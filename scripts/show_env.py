# --- Example Usage (for testing the environment) ---
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import tyro
from rich.console import Console
from rich.table import Table

from balance.env import BehaviorSettings, ResetSettings, SegwayEnv, SimulationSettings
from balance.utils import load_robot_model


@dataclass
class ShowEnvSettings:
    sim: SimulationSettings
    playback_speed: float = 0.2  # Playback speed for the simulation

    @property
    def wall_clock_timestep(self):
        return self.sim.robot_timestep / self.playback_speed


@dataclass
class Settings:
    view: ShowEnvSettings
    behavior: BehaviorSettings
    reset: ResetSettings


class ShowEnv:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.console = Console()

    def run(self):
        # Create the environment instance
        model = load_robot_model()
        env = SegwayEnv(model, self.settings.view.sim, self.settings.behavior, self.settings.reset)

        # Reset the environment to get the initial state
        obs, info = env.reset()
        print("Initial Observation:", obs)
        print("Observation Space:", env.observation_space)
        print("Action Space:", env.action_space)

        # Optional: Launch viewer manually for testing
        viewer = mujoco.viewer.launch_passive(env._model, env._model_data)
        viewer.cam.distance = 3.0

        env.set_movement_commands(
            speed=(env.np_random.uniform(-1, 1)),
            turn=(env.np_random.uniform(-1, 1))
        )

        try:
            self.env_step(env, viewer)
        except KeyboardInterrupt:
            pass
        finally:
            env.close()  # Close the viewer cleanly
            if viewer is not None:
                viewer.close()


    def env_step(self, env: SegwayEnv, viewer: mujoco.viewer.Handle):
        while viewer.is_running():
            viewer.speed = self.settings.view.playback_speed
            step_start = time.time()

            # In a real training loop, action would come from the agent:
            # action = agent.predict(obs)
            # For testing, use a fixed action or sample from action space
            action = env.action_space.sample()  # Example: random actions
            # action = test_action # Example: fixed action

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            self.print_update(reward, info)

            if viewer is not None:
                viewer.sync()  # Sync viewer with simulation data

            # Check if episode is done
            if terminated or truncated:
                print("Episode finished.")
                obs, info = env.reset()  # Reset for a new episode
                print("Resetting environment.")

            # Optional: Add sleep to match wall-clock time if not using viewer.sync()
            self.sleep_until_next_step(step_start)


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



if __name__ == "__main__":
    settings = tyro.cli(Settings)
    ShowEnv(settings).run()
