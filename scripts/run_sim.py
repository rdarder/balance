from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import tyro

from balance.obs_action_recording import ImuActionEpisodeRecorder, NullEpisodeRecoder
from tf_agents.specs import array_spec
from tf_agents.trajectories import TimeStep, PolicyStep
import tensorflow as tf

from balance.env import (
    BehaviorSettings,
    ResetSettings,
    SegwayEnv,
    SimulationSettings,
)
from balance.utils import load_robot_model
from balance.observation_processing import EncoderWrapper, WorldModelEncoderSettings

@dataclass
class RunSettings:
    """Settings for running the simulation."""
    sim: SimulationSettings
    model_checkpoints_dir: str = "ppo_training_results/policy/"
    use_random_policy: bool = False

    # Visualization & Playback
    headless: bool = False
    playback_speed: float = 1.0

    # Data Recording
    record_data: bool = False # If True, record (obs, prev_action) sequences
    output_dir: str = "simulation_data" # Directory to save recorded data
    min_episode_length: int = 50 # Min steps for an episode to be saved if recording

    # --- Noise Injection ---
    obs_noise_scale: float = 0.0 # Stddev of Gaussian noise added to observations fed to the policy
    action_noise_scale: float = 0.0 # Stddev of Gaussian noise added to the policy's output action

    # Run Control
    num_episodes: int = 10

    @property
    def wall_clock_timestep(self):
        return self.sim.robot_timestep / self.playback_speed

    def __post_init__(self):
        if self.record_data:
            self.output_path = Path(self.output_dir)
        else:
            self.output_path = None
        if self.obs_noise_scale < 0.0:
            raise ValueError("obs_noise_scale must be non-negative.")
        if self.action_noise_scale < 0.0:
            raise ValueError("action_noise_scale must be non-negative.")

@dataclass
class Settings:
    """Overall settings combining run, behavior, reset, and world model."""
    run: RunSettings
    behavior: BehaviorSettings
    reset: ResetSettings
    world_model: WorldModelEncoderSettings # Add world model settings


class ActionAdapter:
    def action(self, time_step: TimeStep) -> np.ndarray:
        raise NotImplementedError


class RandomAction(ActionAdapter):
    def action(self, time_step: TimeStep) -> np.ndarray:
        return np.random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)


class TfActionPlaybackAdapter(ActionAdapter):
    def __init__(self, tf_policy):
        self.tf_policy = tf_policy

    def action(self, time_step: TimeStep) -> PolicyStep:
        # Ensure observation has the expected dtype for the policy
        # This might be important if the wrapper changes dtype, though unlikely here.
        # obs_dtype = self.tf_policy.time_step_spec.observation.dtype # Get expected dtype
        # time_step = time_step._replace(
        #     observation=tf.cast(time_step.observation, obs_dtype)
        # )

        batched_time_step = tf.nest.map_structure(
            lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=t.dtype), 0),
            time_step
        )
        # The loaded policy MUST be compatible with the observation spec
        # (raw or latent) provided by the potentially wrapped environment.
        action_step = self.tf_policy.action(batched_time_step)
        action = action_step.action.numpy()[0]
        return action

class SimulationRunner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self):
        print("Initializing environment...")
        mujoco_model = load_robot_model()
        mujoco_model_data = mujoco.MjData(mujoco_model)
        env = SegwayEnv(
            mujoco_model,
            mujoco_model_data,
            self.settings.run.sim,
            self.settings.behavior,
            self.settings.reset
        )

        # --- Conditionally Wrap Environment ---
        if self.settings.world_model.use_encoder:
            print("Wrapping environment with world model encoder for simulation.")
            try:
                env = EncoderWrapper(
                    environment=env, # Pass the base env
                    encoder_checkpoint_path=self.settings.world_model.checkpoint_path,
                    latent_dim=self.settings.world_model.latent_dim,
                    input_features=self.settings.world_model.input_features
                )
                print(f"Using EncoderWrapper. Final Observation Spec: {env.observation_spec()}")
            except Exception as e:
                print(f"FATAL: Failed to initialize EncoderWrapper: {e}")
                # Decide how to handle failure: fallback or exit? Exit for safety.
                raise e
        else:
            print("Using raw observations from environment for simulation.")
            print(f"Observation Spec: {env.observation_spec()}")
        # ------------------------------------

        print("Loading policy adapter...")
        # Get action spec from the potentially wrapped env
        env_action_spec = env.action_spec()
        policy = self._load_policy_adapter(env_action_spec)

        viewer = None
        if not self.settings.run.headless:
            print("Launching viewer...")
            viewer = mujoco.viewer.launch_passive(mujoco_model, mujoco_model_data)
            if viewer:
                 viewer.speed = self.settings.run.playback_speed
            else:
                print("Warning: Failed to launch viewer.")
                self.settings.run.headless = True

        if self.settings.run.record_data:
            recorder = ImuActionEpisodeRecorder(self.settings.run.output_path, self.settings.run.min_episode_length)
        else:
            recorder = NullEpisodeRecoder()

        try:
            for i in range(self.settings.run.num_episodes):
                print(f"--- Starting Episode {i+1}/{self.settings.run.num_episodes} ---")
                # env.reset() will call EncoderWrapper._reset() if wrapped
                time_step = env.reset()

                while not time_step.is_last():
                    if viewer and not viewer.is_running():
                        print("Viewer closed by user.")
                        return # Exit cleanly

                    step_start = time.time()
                    action = policy.action(time_step)
                    time_step = env.step(action)
                    if viewer:
                        viewer.sync()
                        self.sleep_until_next_step(step_start)
                    recorder.record_step(time_step, action)
            recorder.finalize()

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Exiting.")
        finally:
            if viewer and viewer.is_running():
                print("Closing viewer.")
                viewer.close()

    def _load_policy_adapter(self, env_action_spec):
        base_adapter: ActionAdapter

        if self.settings.run.use_random_policy:
            print("Using Random Policy as base.")
            base_adapter = RandomAction()
        else:
            try:
                policy_dir = find_latest_checkpoint(self.settings.run.model_checkpoints_dir)
                print(f"Loading saved policy from: {policy_dir}")
                # IMPORTANT: The loaded policy's expected observation spec MUST match
                # the observation spec of the (potentially wrapped) environment.
                # If use_encoder=True, the policy MUST have been trained with the encoder.
                # If use_encoder=False, the policy MUST have been trained on raw obs.
                saved_tf_policy = tf.saved_model.load(policy_dir)
                # TODO: Add check: saved_tf_policy.time_step_spec.observation == env.observation_spec() ?
                base_adapter = TfActionPlaybackAdapter(saved_tf_policy)
            except (FileNotFoundError, Exception) as e:
                print(f"Error loading saved policy: {e}")
                print("Falling back to Random Policy as base.")
                base_adapter = RandomAction()

        obs_noise = self.settings.run.obs_noise_scale
        act_noise = self.settings.run.action_noise_scale

        if obs_noise > 0.0 or act_noise > 0.0:
            print(f"Wrapping base adapter with noise (Obs: {obs_noise}, Act: {act_noise}).")
            noisy_adapter = NoisyActionAdapter(
                base_adapter=base_adapter,
                obs_noise_scale=obs_noise,
                action_noise_scale=act_noise,
                action_spec=env_action_spec
            )
            return noisy_adapter
        else:
            print("No noise configured, using base adapter directly.")
            return base_adapter


    def sleep_until_next_step(self, step_started_at):
        if not self.settings.run.headless:
            elapsed_since_robot_step = time.time() - step_started_at
            time_until_next_robot_step = (self.settings.run.wall_clock_timestep -
                                          elapsed_since_robot_step)
            if time_until_next_robot_step > 0:
                time.sleep(time_until_next_robot_step)


def find_latest_checkpoint(path: str) -> str:
    """Given a path, find which directory in it has the latest checkpoint.
    List all directories within path that contain a file named saved_model.pb.
    Return the one that's most recent. Uses pathlib.
    """
    path = Path(path).expanduser()
    candidates = []
    if not path.is_dir():
         raise FileNotFoundError(f"Policy directory not found: {path}")

    for d in path.iterdir():
        if d.is_dir() and (d / "saved_model.pb").is_file():
            candidates.append(d)

    if not candidates:
        raise FileNotFoundError(f"No policy directories with 'saved_model.pb' found in: {path}")

    return str(max(candidates, key=lambda p: p.stat().st_mtime))


class NoisyActionAdapter(ActionAdapter):
    """Wraps another ActionAdapter to add noise to observations and actions."""

    def __init__(self,
                 base_adapter: ActionAdapter,
                 obs_noise_scale: float,
                 action_noise_scale: float,
                 action_spec: array_spec.BoundedArraySpec):
        if not isinstance(base_adapter, ActionAdapter):
             raise TypeError("base_adapter must be an instance of ActionAdapter")
        if obs_noise_scale < 0.0 or action_noise_scale < 0.0:
             raise ValueError("Noise scales must be non-negative.")

        self.base_adapter = base_adapter
        self.obs_noise_scale = obs_noise_scale
        self.action_noise_scale = action_noise_scale
        self.action_spec = action_spec
        print(f"NoisyActionAdapter initialized with obs_noise={obs_noise_scale}, action_noise={action_noise_scale}")


    def action(self, time_step: TimeStep) -> np.ndarray:
        """Gets action from base adapter with noisy inputs/outputs."""

        noisy_observation = time_step.observation
        if self.obs_noise_scale > 0.0:
            obs_noise = np.random.normal(
                loc=0.0,
                scale=self.obs_noise_scale,
                size=time_step.observation.shape
            ).astype(time_step.observation.dtype)
            noisy_observation = time_step.observation + obs_noise
            # Note: We are NOT clipping the noisy observation here, assuming the policy
            # should be robust enough or that noise scale is reasonable. Clipping
            # observations can sometimes mask issues or introduce bias.

        # Create the TimeStep potentially with noisy observation
        if self.obs_noise_scale > 0.0:
            # should use copy.replace when on python >= 3.13
            noisy_input_time_step = time_step._replace(observation=noisy_observation)
        else:
            noisy_input_time_step = time_step

        base_action = self.base_adapter.action(noisy_input_time_step)

        if not isinstance(base_action, np.ndarray):
             # This might happen if a base adapter returns something else unexpectedly
             # Convert or handle as needed. For now, assume it's ndarray.
             raise TypeError("Unknown type {type(base_action)} for base_action.")

        noisy_action = base_action
        if self.action_noise_scale > 0.0:
            action_noise = np.random.normal(
                loc=0.0,
                scale=self.action_noise_scale,
                size=base_action.shape
            ).astype(base_action.dtype)
            noisy_action = base_action + action_noise

        clipped_noisy_action = np.clip(
            noisy_action,
            self.action_spec.minimum,
            self.action_spec.maximum
        )

        return clipped_noisy_action.astype(self.action_spec.dtype)


if __name__ == "__main__":
    settings_from_cmdline = tyro.cli(Settings)
    runner = SimulationRunner(settings_from_cmdline)
    runner.run()