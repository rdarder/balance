from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import typing

import numpy as np
import mujoco
import mujoco.viewer
import tyro

from balance.obs_action_recording import ImuActionEpisodeRecorder, NullEpisodeRecoder
from tf_agents.specs import array_spec, BoundedTensorSpec, TensorSpec
from tf_agents.trajectories import TimeStep
import tensorflow as tf

from tf_agents.policies import actor_policy, TFPolicy
from tf_agents.typing import types
from balance.rl_model import create_ppo_networks

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
    model_checkpoints_dir: str = "ppo_training_results/train/policy/"
    use_random_policy: bool = False

    # Visualization & Playback
    headless: bool = False
    playback_speed: float = 1.0

    # Data Recording
    record_data: bool = False
    output_dir: str = "simulation_data"
    min_episode_length: int = 50

    # Noise Injection
    obs_noise_scale: float = 0.0
    action_noise_scale: float = 0.0

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
    world_model: WorldModelEncoderSettings


class ActionAdapter:
    """Abstract base class for policy adapters."""
    def action(self, time_step: TimeStep) -> np.ndarray:
        raise NotImplementedError


class RandomAction(ActionAdapter):
    """Adapter that returns random actions."""
    def action(self, time_step: TimeStep) -> np.ndarray:
        # Assuming action spec is always (-1, 1) for shape (2,)
        return np.random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)


class TfActionPlaybackAdapter(ActionAdapter):
    """Adapter that uses a TF-Agents policy for actions."""
    def __init__(self, tf_policy: TFPolicy):
        self.tf_policy = tf_policy

    def action(self, time_step: TimeStep) -> np.ndarray:
        # Batch the time_step for the policy
        batched_time_step = tf.nest.map_structure(
            lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=t.dtype), 0),
            time_step
        )
        action_step = self.tf_policy.action(batched_time_step)
        return action_step.action.numpy()[0]


class NoisyActionAdapter(ActionAdapter):
    """Wraps another ActionAdapter to add noise to observations and/or actions."""
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
        # print(f"NoisyActionAdapter initialized with obs_noise={obs_noise_scale}, action_noise={action_noise_scale}")

    def action(self, time_step: TimeStep) -> np.ndarray:
        """Gets action from base adapter with potentially noisy inputs/outputs."""
        noisy_observation = time_step.observation
        if self.obs_noise_scale > 0.0:
            obs_noise = np.random.normal(
                loc=0.0,
                scale=self.obs_noise_scale,
                size=time_step.observation.shape
            ).astype(time_step.observation.dtype)
            noisy_observation = time_step.observation + obs_noise

        # Create the TimeStep potentially with noisy observation
        noisy_input_time_step = time_step._replace(observation=noisy_observation)

        base_action = self.base_adapter.action(noisy_input_time_step)

        if not isinstance(base_action, np.ndarray):
             raise TypeError(f"Base adapter returned unexpected type {type(base_action)}")

        noisy_action = base_action
        if self.action_noise_scale > 0.0:
            action_noise = np.random.normal(
                loc=0.0,
                scale=self.action_noise_scale,
                size=base_action.shape
            ).astype(base_action.dtype)
            noisy_action = base_action + action_noise

        # Clip final action to spec bounds
        clipped_noisy_action = np.clip(
            noisy_action,
            self.action_spec.minimum,
            self.action_spec.maximum
        )
        return clipped_noisy_action.astype(self.action_spec.dtype)


class SimulationRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.env: typing.Optional[SegwayEnv | EncoderWrapper] = None # For holding the env instance

    def _load_tf_policy_from_checkpoint(
        self,
        ckpt_dir: str,
        # Accept PyEnvironment Specs (ArraySpec)
        py_time_step_spec: TimeStep,
        py_action_spec: types.NestedArraySpec,
        py_observation_spec: types.NestedArraySpec
    ) -> actor_policy.ActorPolicy:
        """Loads the latest TF-Agents policy checkpoint."""
        print(f"Looking for latest checkpoint in: {ckpt_dir}")
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if not latest_ckpt:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        print(f"Found latest checkpoint prefix: {latest_ckpt}")

        tensor_observation_spec = TensorSpec.from_spec(py_observation_spec)
        tensor_action_spec = BoundedTensorSpec.from_spec(py_action_spec)
        tensor_time_step_spec = tf.nest.map_structure(
            TensorSpec.from_spec, py_time_step_spec
        )

        # Re-create the network structure using TensorSpecs
        # TODO: we must ensure this matches the training shapes.
        actor_fc_layers = (32, 32)
        value_fc_layers = (64, 64) # Needed for structure, not weights

        actor_net, _ = create_ppo_networks(
            tensor_observation_spec, # Use TensorSpec
            tensor_action_spec,      # Use TensorSpec
            actor_fc_layers=actor_fc_layers,
            value_fc_layers=value_fc_layers,
        )

        # Instantiate the policy using TensorSpecs
        tf_policy = actor_policy.ActorPolicy(
            time_step_spec=tensor_time_step_spec, # Use TensorSpec
            action_spec=tensor_action_spec,       # Use TensorSpec
            actor_network=actor_net,
            training=False # Set to False for inference
        )


        # Create a checkpoint object mapping the name used during saving
        # to the policy object we just created.
        ckpt = tf.train.Checkpoint(policy=tf.train.Checkpoint(
            _wrapped_policy=tf.train.Checkpoint(_actor_network=actor_net)))

        status = ckpt.restore(latest_ckpt).expect_partial()

        try:
            # status.assert_nontrivial_match()
            status.assert_existing_objects_matched()
            print("Checkpoint restore reported a non-trivial match.")
        except AssertionError as e:
            print(f"WARNING: Checkpoint restore failed assert_nontrivial_match: {e}")
            raise
            # This suggests nothing or very little matched the policy structure.


        # Optionally add: status.assert_existing_objects_matched() for stricter checks
        print("Policy weights restored from checkpoint.")
        return tf_policy

    def run(self):
        print("Initializing environment...")
        mujoco_model = load_robot_model()
        mujoco_model_data = mujoco.MjData(mujoco_model)
        base_env = SegwayEnv(
            mujoco_model,
            mujoco_model_data,
            self.settings.run.sim,
            self.settings.behavior,
            self.settings.reset
        )

        # --- Conditionally Wrap Environment ---
        if self.settings.world_model.use_encoder:
            print("Wrapping environment with world model encoder for simulation.")
            self.env = EncoderWrapper(
                environment=base_env,
                encoder_checkpoint_path=self.settings.world_model.checkpoint_path,
                latent_dim=self.settings.world_model.latent_dim,
                input_features=self.settings.world_model.input_features
            )
        else:
            print("Using raw observations from environment for simulation.")
            self.env = base_env
        print(f"Final Observation Spec: {self.env.observation_spec()}")
        # ------------------------------------

        # --- Load Policy or Use Random ---
        base_adapter: ActionAdapter
        if self.settings.run.use_random_policy:
            print("Using Random Policy.")
            base_adapter = RandomAction()
        else:
            print("Loading policy from checkpoint...")
            tf_policy = self._load_tf_policy_from_checkpoint(
                ckpt_dir=self.settings.run.model_checkpoints_dir,
                py_time_step_spec=self.env.time_step_spec(),
                py_action_spec=self.env.action_spec(),
                py_observation_spec=self.env.observation_spec()
            )
            base_adapter = TfActionPlaybackAdapter(tf_policy)
        # ---------------------------------

        # --- Apply Noise Wrapper if configured ---
        policy: ActionAdapter = base_adapter # Start with the base
        obs_noise = self.settings.run.obs_noise_scale
        act_noise = self.settings.run.action_noise_scale
        if obs_noise > 0.0 or act_noise > 0.0:
            print(f"Wrapping policy with noise (Obs: {obs_noise}, Act: {act_noise}).")
            policy = NoisyActionAdapter(
                base_adapter=base_adapter,
                obs_noise_scale=obs_noise,
                action_noise_scale=act_noise,
                action_spec=self.env.action_spec() # Use final env's action spec
            )
        # -----------------------------------------

        # --- Initialize Viewer ---
        viewer = None
        if not self.settings.run.headless:
            print("Launching viewer...")
            viewer = mujoco.viewer.launch_passive(mujoco_model, mujoco_model_data)
            viewer.speed = self.settings.run.playback_speed

        if self.settings.run.record_data:
            recorder = ImuActionEpisodeRecorder(self.settings.run.output_path, self.settings.run.min_episode_length)
        else:
            recorder = NullEpisodeRecoder()

        try:
            for i in range(self.settings.run.num_episodes):
                time_step = self.env.reset()

                total_reward = 0
                steps = 0

                while not time_step.is_last():
                    if viewer and not viewer.is_running():
                        print("Viewer closed by user.")
                        return # Exit cleanly

                    step_start = time.time()
                    action = policy.action(time_step)
                    time_step = self.env.step(action)
                    total_reward += time_step.reward
                    steps +=1
                    if viewer:
                        viewer.sync()
                        self.sleep_until_next_step(step_start)
                    recorder.record_step(time_step, action)
                print(f"--- Finished Episode {i+1}/{self.settings.run.num_episodes}. Reward: "
                      f"{total_reward} Steps: {steps}---")

            recorder.finalize()

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Exiting.")
            recorder.finalize() # Attempt to finalize recording on interrupt
        finally:
            if viewer and viewer.is_running():
                print("Closing viewer.")
                viewer.close()
            self.env.close() # Close the environment
        # ---------------------

    def sleep_until_next_step(self, step_started_at):
        """Sleeps to maintain the desired playback speed."""
        if not self.settings.run.headless:
            elapsed_since_robot_step = time.time() - step_started_at
            time_until_next_robot_step = (self.settings.run.wall_clock_timestep -
                                          elapsed_since_robot_step)
            if time_until_next_robot_step > 0:
                time.sleep(time_until_next_robot_step)


if __name__ == "__main__":
    settings_from_cmdline = tyro.cli(Settings)
    runner = SimulationRunner(settings_from_cmdline)
    runner.run()