# Segway Balancing Robot - RL Control (TF-Agents) & ESP32 Deployment

This project explores training Reinforcement Learning (RL) agents using TF-Agents to control a
simulated Segway-like robot in MuJoCo. The primary goal is to develop a control policy capable of
balancing the robot and eventually deploy this policy onto an ESP32-C3 microcontroller for real-time
operation.

## Key Features

* **MuJoCo Simulation:** A custom `SegwayEnv` environment built on
  `tf_agents.environments.PyEnvironment` simulating a two-wheeled balancing robot (
  `balance/env.py`).
* **RL Training:** Uses the PPO (Proximal Policy Optimization) algorithm from TF-Agents for
  training (`scripts/train_rl.py`).
    * Supports parallel environment training for faster data collection.
    * Includes standard training features like checkpointing, evaluation loops, and TensorBoard
      logging.
    * Configurable via `tyro` for hyperparameters and settings.
* **Policy Simulation:** A script (`scripts/run_sim.py`) to load trained policy checkpoints and run
  them in the MuJoCo simulation.
    * Includes visualization using `mujoco.viewer`.
    * Optionally records simulation data (observations, actions) for further analysis or world model
      training.
    * Supports injecting noise into observations or actions for robustness testing.
* **World Model Framework (Experimental):** Includes an `EncoderWrapper` and settings (
  `balance/observation_processing.py`) to facilitate future work on training a world model (e.g.,
  GRU) and a policy based on its latent states.
* **Deployment Target:** Aims for deployment on an ESP32-C3 microcontroller using TensorFlow Lite
  for Microcontrollers.

## Current Status

* Successfully trains a PPO policy using raw IMU sensor data + desired commands as observations (
  `normalize_observations=False` in PPOAgent).
* The simulation script (`run_sim.py`) correctly loads checkpoints from this training setup and runs
  the policy in the environment.
* The complex checkpoint loading process involving TF-Agents internal wrappers has been debugged and
  verified for the non-normalized agent setup.

## Getting Started

1. **Dependencies:** Ensure you have Python 3.x, MuJoCo, TensorFlow, TF-Agents, NumPy, SciPy, and
   Tyro installed.
    