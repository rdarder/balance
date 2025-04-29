# --- Example Usage (for testing the environment) ---
import time
import mujoco
import mujoco.viewer
import numpy as np

from balance.env import SegwayEnv
from balance.utils import load_robot_model


def show_env():
    # Create the environment instance
    model = load_robot_model()
    env = SegwayEnv(model)

    # Reset the environment to get the initial state
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    # Optional: Launch viewer manually for testing
    viewer = mujoco.viewer.launch_passive(env._model, env._model_data)

    # Simple loop to test stepping the environment
    # Apply a constant forward torque for a few steps
    test_action = np.array(
        [0.5, 0.5], dtype=np.float32
    )  # Example: apply half max torque forward
    # Or set desired commands if you want to test that part of the observation
    env.set_movement_commands(speed=0.8, turn=0.0)
    # test_action = np.array([0.0, 0.0], dtype=np.float32) # Agent would learn to use these

    running = True
    try:
        while running and viewer.is_running():
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
            time_until_next_step = env._timestep * env._frame_skip - (
                time.time() - step_start
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()  # Close the viewer cleanly
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    show_env()
