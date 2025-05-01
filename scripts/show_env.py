from __future__ import annotations

import time
from dataclasses import dataclass
import os

import numpy as np
import mujoco
import mujoco.viewer
import tyro
from rich.console import Console
from rich.table import Table
from tf_agents.trajectories import TimeStep, PolicyStep
import tensorflow as tf

from balance.env import BehaviorSettings, ResetSettings, SegwayEnv, SimulationSettings
from balance.utils import load_robot_model

@dataclass
class ShowEnvSettings:
    sim: SimulationSettings
    playback_speed: float = 1.0  # Playback speed for the simulation
    model_checkpoints_dir: str = "ppo_training_results/policy/"
    use_random_policy: bool = (
            False # When true, use a random policy instead of loading a checkpoint.
    )
    num_episodes: int = 10
    @property
    def wall_clock_timestep(self):
        return self.sim.robot_timestep / self.playback_speed


@dataclass
class Settings:
    view: ShowEnvSettings
    behavior: BehaviorSettings
    reset: ResetSettings


class ActionAdapter:
    def action(self, time_step: TimeStep) ->  np.ndarray:
        raise NotImplementedError


class RandomAction(ActionAdapter):
    def action(self, time_step: TimeStep) -> np.ndarray:
        return np.random.normal(0.0, 1.0, size=(2,))


class TfActionPlaybackAdapter(ActionAdapter):
    def __init__(self, tf_policy):
        self.tf_policy = tf_policy

    def action(self, time_step: TimeStep) -> PolicyStep:
        batched_time_step = tf.nest.map_structure(
            lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=t.dtype), 0),
            time_step
        )
        action_step = self.tf_policy.action(batched_time_step)
        action = action_step.action.numpy()[0]
        print(action)
        return action


class ShowEnv:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.console = Console()

    def run(self):
        # Create the environment instance
        mujoco_model = load_robot_model()
        env = SegwayEnv(mujoco_model, self.settings.view.sim, self.settings.behavior, self.settings.reset)

        viewer = mujoco.viewer.launch_passive(env._model, env._model_data)
        policy = self._load_policy_adapter()

        try:
            for i in range(settings.view.num_episodes):
                time_step = env.reset()
                while not time_step.is_last():
                    if not viewer.is_running():
                        return
                    viewer.speed = self.settings.view.playback_speed
                    step_start = time.time()
                    action = policy.action(time_step)
                    time_step = env.step(action)
                    viewer.sync()
                    self.sleep_until_next_step(step_start)
        finally:
            env.close()  # Close the viewer cleanly
            viewer.close()

    def _load_policy_adapter(self):
        if self.settings.view.use_random_policy:
            return RandomAction()
        else:
            policy_dir = find_latest_checkpoint(settings.view.model_checkpoints_dir)
            saved_tf_policy = tf.saved_model.load(policy_dir)
            return TfActionPlaybackAdapter(saved_tf_policy)


    def print_update(self, reward: float, info: dict):

        table = Table(title="Step Information")
        table.add_column("Reward")
        table.add_column("Value")
        table.add_column("Target")

        table.add_row(
            fmt(reward), fmt(info['axle_angle_rad']), fmt(settings.behavior.max_standing_up_roll)
        )

        self.console.clear()
        self.console.print(table)

    def sleep_until_next_step(self, step_started_at):
        elapsed_since_robot_step = time.time() - step_started_at
        time_until_next_robot_step = (self.settings.view.wall_clock_timestep -
                                      elapsed_since_robot_step)
        if time_until_next_robot_step > 0:
            time.sleep(time_until_next_robot_step)

def fmt(n, places: int = 4):
    if isinstance(n, int):
        return str(n) + ' ' * (places+1)
    elif isinstance(n, float):
        return f"{n:.{places}f}"
    else:
        raise NotImplementedError(f"'{type(n)}' not implemented")

def find_latest_checkpoint(path: str):
    """Given a path, find which directory in it has the latest checkpoint.
    List all directories within path that contain a file named saved_mode.pb.
    Return the one that's most recent
    """
    candidates = []
    for d in os.listdir(path):
        full_path = os.path.join(path, d)
        if os.path.isdir(full_path) and "saved_model.pb" in os.listdir(full_path):
            candidates.append(full_path)
    if not candidates:
        raise FileNotFoundError(f"No policy directories with 'saved_model.pb' found in: {path}")
    return max(candidates, key=os.path.getmtime)



if __name__ == "__main__":
    settings = tyro.cli(Settings)
    ShowEnv(settings).run()
