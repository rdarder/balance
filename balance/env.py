import gymnasium as gym
import numpy as np
import mujoco

from balance.checks import check_state


class SegwayEnv(gym.Env):
    """
    A custom Gymnasium environment for the MuJoCo Segway model.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        timestep=0.002,
        frame_skip=20,
    ):
        super().__init__()
        self._model = model
        self._model_data = mujoco.MjData(self._model)

        # Set simulation options (can override XML)
        self._timestep = timestep
        self._frame_skip = frame_skip  # Number of simulation steps per environment step
        self._model.opt.timestep = timestep

        # Find actuator IDs
        self._left_motor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left-motor"
        )
        self._right_motor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right-motor"
        )
        check_state(self._left_motor_id != -1, "left motor not found in model")
        check_state(self._right_motor_id != -1, "right motor not found in model")

        # Find sensor IDs
        self._imu_accel_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel"
        )
        check_state(self._imu_accel_id != -1, "IMU accel not found in model")
        self._imu_gyro_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
        )
        check_state(self._imu_gyro_id != -1, "IMU gyro not found in model")

        # Store initial state for reset (optional, mj_resetData is standard)
        self._initial_qpos = np.copy(self._model_data.qpos)
        self._initial_qvel = np.copy(self._model_data.qvel)

        # --- Environment Spaces ---
        # Action space: [left_motor_pwm_duty_cycle, right_motor_pwm_duty_cycle]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation space: [imu_accel (3), imu_gyro (3), desired_speed (1), desired_turn (1)]
        # Note: Bounds for IMU are technically unbounded, desired commands are [-1, 1]
        # We use -inf/inf for simplicity of the Box space definition, but be mindful of this.
        obs_dim = 6 + 2  # IMU (accel+gyro) + desired_speed + desired_turn
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Internal State for Desired Commands ---
        # These represent the commands the agent is trying to follow.
        # They need to be set externally (e.g., by the training loop)
        self._desired_speed = 0.0
        self._desired_turn = 0.0

        # --- Viewer (Optional) ---
        self.viewer = None

    def set_movement_commands(self, speed: float, turn: float):
        """Sets the desired speed and turn commands for the agent to follow."""
        # Clamp commands to [-1, 1] as per your description
        self._desired_speed = np.clip(speed, -1.0, 1.0)
        self._desired_turn = np.clip(turn, -1.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        """Collects the current observation."""
        # Get IMU data from sensors
        imu_accel = self._model_data.sensordata[
            self._imu_accel_id : self._imu_accel_id + 3
        ]
        imu_gyro = self._model_data.sensordata[
            self._imu_gyro_id : self._imu_gyro_id + 3
        ]

        # Concatenate IMU data with desired commands
        obs = np.concatenate(
            [
                imu_accel,
                imu_gyro,
                [self._desired_speed],  # Include desired speed
                [self._desired_turn],  # Include desired turn
            ]
        ).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the MuJoCo simulation data
        mujoco.mj_resetData(self._model, self._model_data)

        # Optional: Perturb initial state slightly for robustness
        # e.g., self.data.qpos += self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        # mujoco.mj_forward(self.model, self.data) # Need to call forward after changing qpos/qvel

        # Reset desired commands (e.g., to zero, or sample a new target)
        self._desired_speed = self.np_random.uniform(-1, 1)
        self._desired_turn = self.np_random.uniform(-1, 1)

        # Get the initial observation
        observation = self._get_obs()

        # Return observation and info dictionary (standard for Gymnasium)
        info = {}  # Can include debugging info here
        return observation, info

    def step(self, action):
        # Ensure action is within bounds (RL algorithms usually handle this, but good practice)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply the action to the motor controls
        self._model_data.ctrl[self._left_motor_id] = action[0]
        self._model_data.ctrl[self._right_motor_id] = action[1]

        # --- Simulate physics ---
        # Run mj_step multiple times for frame skipping
        for _ in range(self._frame_skip):
            mujoco.mj_step(self._model, self._model_data)
            # Optional: Check for termination/truncation *within* frame skip if needed
            # e.g., if self._is_terminated(): break

        # --- Get next observation ---
        observation = self._get_obs()

        # --- Calculate reward ---
        # THIS IS WHERE YOU DEFINE YOUR REWARD FUNCTION
        # Access simulator state via self.data
        # Examples:
        # chassis_height = self.data.qpos[2] # Assuming free joint qpos is [x, y, z, qw, qx, qy, qz]
        # chassis_orientation = self.data.qpos[3:7] # Quaternion
        # chassis_angular_vel = self.data.qvel[3:6] # Angular velocity
        # chassis_linear_vel = self.data.qvel[0:3] # Linear velocity
        # wheel_vel_left = self.data.qvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left-wheel-joint")]
        # wheel_vel_right = self.data.qvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right-wheel-joint")]

        reward = 0.0  # Placeholder: Implement your reward logic here

        # --- Check for termination or truncation ---
        # Define conditions for ending an episode
        # Examples:
        # - Robot falls over (chassis angle too large)
        # - Chassis height too low
        # - Simulation time exceeds a limit (truncation)

        terminated = (
            False  # Set to True if the episode ends due to failure (e.g., falling)
        )
        truncated = False  # Set to True if the episode ends due to time limit or other non-failure reason

        # Example termination condition (falling over):
        # Get chassis orientation (quaternion)
        chassis_quat = self._model_data.qpos[3:7]
        # Convert quaternion to Euler angles or check vertical vector
        # A simple check: is the Z-axis of the chassis pointing mostly up?
        # Get the Z-axis vector in world coordinates from the chassis body's orientation
        # This requires accessing the body's orientation matrix, which is in data.xmat
        chassis_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
        )
        # if chassis_body_id != -1:
        #     chassis_z_axis_world = self._model_data.xmat[chassis_body_id].reshape(3, 3)[
        #         :, 2
        #     ]
        #     # Check if the dot product with world Z-axis (0,0,1) is below a threshold
        #     # A dot product of 1 means perfectly upright, 0 means horizontal, -1 means upside down
        #     upright_threshold = 0.5  # Example: roughly 60 degrees tilt
        #     if chassis_z_axis_world[2] < upright_threshold:
        #         terminated = True

        # Example truncation condition (time limit):
        # max_episode_steps = 500 # Define this in __init__ or as a parameter
        # if self.data.time >= max_episode_steps * self.model.opt.timestep * self.frame_skip:
        #     truncated = True

        # --- Info dictionary ---
        info = {}  # Can add debugging info, e.g., info = {"chassis_height": chassis_height}

        return observation, reward, terminated, truncated, info

    def close(self):
        """Closes the viewer if it's open."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # Optional: Add a render method if you want to use gym.make(..., render_mode='human')
    # def render(self):
    #     if self.viewer is None:
    #         self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    #     self.viewer.sync()
