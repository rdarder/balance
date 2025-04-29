# --- Example Usage (for testing the environment) ---
import time
from dataclasses import dataclass

import mujoco
import tyro
import mujoco.viewer
import numpy as np

from balance.env import SegwayEnv, SimulationSettings, BehaviorSettings
from balance.utils import load_robot_model


@dataclass
class ShowEnvSettings:
    sim: SimulationSettings
    playback_speed: float = 0.2  # Playback speed for the simulation
    episode_length: float = (
        0.8  # How long until resetting the episode to start a new one.
    )

    @property
    def max_robot_steps(self):
        return int(self.episode_length / self.sim.robot_timestep)

    @property
    def wall_clock_timestep(self):
        return self.sim.robot_timestep / self.playback_speed


def show_env(view: ShowEnvSettings, behavior: BehaviorSettings):
    # Create the environment instance
    model = load_robot_model()
    env = SegwayEnv(model, view.sim, behavior)

    # Reset the environment to get the initial state
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    # Optional: Launch viewer manually for testing
    viewer = mujoco.viewer.launch_passive(env._model, env._model_data)
    viewer.cam.distance = 3.0

    # Simple loop to test stepping the environment
    # Apply a constant forward torque for a few steps
    test_action = np.array(
        [0.5, 0.5], dtype=np.float32
    )  # Example: apply half max torque forward
    # Or set desired commands if you want to test that part of the observation
    env.set_movement_commands(speed=0.8, turn=0.0)
    # test_action = np.array([0.0, 0.0], dtype=np.float32) # Agent would learn to use these

    running = True
    # --- Check for truncation ---

    try:
        while running and viewer.is_running():
            viewer.speed = view.playback_speed
            step_start = time.time()

            # In a real training loop, action would come from the agent:
            # action = agent.predict(obs)
            # For testing, use a fixed action or sample from action space
            action = env.action_space.sample()  # Example: random actions
            # action = test_action # Example: fixed action

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Print some info
            # print(f"Sim Time: {env.data.time:.3f}, Obs: {obs[:6].round(2)}, Reward: {reward:.3f}, Terminated: {terminated}, Truncated: {truncated}")
            # print(f"Applied Ctrl: {env.data.ctrl[env._left_motor_id]:.3f}, {env.data.ctrl[env._right_motor_id]:.3f}")

            if viewer is not None:
                viewer.sync()  # Sync viewer with simulation data

            # Check if episode is done
            if terminated or truncated:
                print("Episode finished.")
                obs, info = env.reset()  # Reset for a new episode
                print("Resetting environment.")
                # Optionally break the loop after one episode for simple testing
                # running = False

            # Optional: Add sleep to match wall-clock time if not using viewer.sync()
            sleep_until_next_step(step_start, view)

            if env.episode_steps >= view.max_robot_steps:
                env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()  # Close the viewer cleanly
        if viewer is not None:
            viewer.close()


def sleep_until_next_step(step_started_at, view):
    elapsed_since_robot_step = time.time() - step_started_at
    time_until_next_robot_step = view.wall_clock_timestep - elapsed_since_robot_step
    if time_until_next_robot_step > 0:
        time.sleep(time_until_next_robot_step)


if __name__ == "__main__":
    app = tyro.cli(show_env)
    app()
