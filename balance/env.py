import gymnasium as gym
import numpy as np
import mujoco
from dataclasses import dataclass
import math  # Import math for trigonometric functions
from scipy.spatial.transform import Rotation as R

from balance.checks import check_state, check_argument


@dataclass
class SimulationSettings:
    robot_hz: float = 50.0
    simulation_hz: float = 500.0
    max_init_pitch: float = 1.8  # Max initial pitch in rads.

    def __post_init__(self):
        check_argument(
            self.simulation_hz > self.robot_hz,
            "Simulation frequency must be greater than robot frequency",
        )
        check_argument(
            abs(self.simulation_hz % self.robot_hz) == 0,
            "Simulation frequency must be a multiple of robot frequency",
        )

    @property
    def sim_timestep(self):
        return 1 / self.simulation_hz

    @property
    def robot_timestep(self):
        return 1 / self.robot_hz

    @property
    def sim_frames_to_robot_frames(self):
        skip = round(self.simulation_hz / self.robot_hz)
        return max(0, skip)


@dataclass
class BehaviorSettings:
    pass


class SegwayEnv(gym.Env):
    """
    A custom Gymnasium environment for the MuJoCo Segway model.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        sim_settings: SimulationSettings,
        behavior_settings: BehaviorSettings,
    ):
        super().__init__()
        self._sim_settings = sim_settings
        self._model = model
        self._model_data = mujoco.MjData(self._model)
        self._sim_settings = sim_settings
        self._behavior = behavior_settings

        # Set simulation options (can override XML)
        self._model.opt.timestep = self._sim_settings.sim_timestep

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

        # --- Environment Spaces ---
        # Action space: [left_motor_pwm_duty_cycle, right_motor_pwm_duty_cycle]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation space: [imu_accel (3), imu_gyro (3), desired_speed (1), desired_turn (1)]
        obs_dim = 6 + 2  # IMU (accel+gyro) + desired_speed + desired_turn
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Internal State for Desired Commands ---
        self._desired_speed = 0.0
        self._desired_turn = 0.0

        # --- Viewer (Optional) ---
        self.viewer = None

        # --- Constants for Reset ---
        # Calculate required Z height based on XML values
        # Wheel Z offset relative to chassis: -0.025
        # Wheel radius: 0.0265
        self._target_init_z = 0.0265 + 0.025  # = 0.0515

        # --- Internal State for Truncation ---
        self.episode_steps = 0

    def set_movement_commands(self, speed: float, turn: float):
        """Sets the desired speed and turn commands for the agent to follow."""
        self._desired_speed = np.clip(speed, -1.0, 1.0)
        self._desired_turn = np.clip(turn, -1.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        """Collects the current observation."""
        imu_accel = self._model_data.sensordata[
            self._imu_accel_id : self._imu_accel_id + 3
        ]
        imu_gyro = self._model_data.sensordata[
            self._imu_gyro_id : self._imu_gyro_id + 3
        ]
        obs = np.concatenate(
            [
                imu_accel,
                imu_gyro,
                [self._desired_speed],
                [self._desired_turn],
            ]
        ).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the MuJoCo simulation data to initial XML state first
        mujoco.mj_resetData(self._model, self._model_data)

        # Set qpos
        self._model_data.qpos[:] = self.get_initial_pose()
        self._model_data.qvel[0:3] = self._get_initial_velocity()
        self._model_data.qvel[3:] = 0.0  # No angular velocity

        # --- Crucial: Forward dynamics ---
        # Apply the new qpos/qvel and compute derived quantities (like sensor readings)
        mujoco.mj_forward(self._model, self._model_data)
        # ---------------------------------

        # Reset desired commands (sample new random targets for the episode)
        self._desired_speed = self.np_random.uniform(-1, 1)
        self._desired_turn = self.np_random.uniform(-1, 1)

        # Get the initial observation based on the new state
        observation = self._get_obs()

        # Reset internal state
        self.episode_steps = 0

        info = {}
        return observation, info

    def _get_initial_velocity(self):
        # Set qvel (velocities) to a random value in the direction of the wheels
        random_speed = self._get_initial_speed()
        yaw_angle = self._model_data.qpos[
            6
        ]  # Yaw angle is the last element of the orientation quaternion
        # Calculate the x and y components of the velocity based on the yaw angle
        vel_x = random_speed * math.cos(yaw_angle)
        vel_y = random_speed * math.sin(yaw_angle)
        vel = vel_x, vel_y, 0.0
        return vel

    def _get_initial_speed(self):
        min_speed = -0.5
        max_speed = 0.5
        random_speed = self.np_random.uniform(min_speed, max_speed)
        return random_speed

    def get_initial_pose(self):
        qpos = np.zeros(self._model.nq)
        qpos[0] = 0.0  # Initial x position
        qpos[1] = 0.0  # Initial y position
        qpos[2] = self._target_init_z  # Set calculated z height
        qpos[3:7] = self.get_initial_orientation()  # Set calculated orientation
        return qpos

    def get_initial_orientation(self):
        # --- Set Random Initial Pose ---
        # 1. Random Pitch Angle (around World Y)
        pitch_angle = self.np_random.uniform(
            -self._sim_settings.max_init_pitch,
            self._sim_settings.max_init_pitch,
        )
        # 2. Random Yaw Angle (around World Z)
        yaw_angle = self.np_random.uniform(-math.pi, math.pi)
        # 3. Roll angle is zero (around World X)
        roll_angle = 0.0

        # --- Calculate Orientation Quaternion using scipy ---
        # Use 'zyx' convention (lowercase = extrinsic): Apply Yaw (Z), then Pitch (Y), then Roll (X)
        # Angles are given in [yaw, pitch, roll] order for 'zyx'
        euler_angles = [yaw_angle, pitch_angle, roll_angle]
        # noinspection PyArgumentList
        rotation = R.from_euler(
            "zyx", euler_angles, degrees=False
        )  # Use False since angles are in radians

        # Get quaternion. Scipy returns in [x, y, z, w] format by default.
        quat_xyzw = rotation.as_quat()

        # Convert to [w, x, y, z] format if needed by your simulation/framework
        orientation = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        return orientation

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._model_data.ctrl[self._left_motor_id] = action[0]
        self._model_data.ctrl[self._right_motor_id] = action[1]

        for _ in range(self._sim_settings.sim_frames_to_robot_frames):
            mujoco.mj_step(self._model, self._model_data)

        self.episode_steps += 1

        observation = self._get_obs()

        # --- Calculate reward ---
        reward = self._get_reward()

        # --- Check for termination or truncation ---
        terminated = self._get_terminated_condition()
        truncated = False

        # Example termination: Check if fallen over
        chassis_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
        )
        info = {}
        return observation, reward, terminated, truncated, info

    def _get_terminated_condition(self):
        return False

    def _get_reward(self):
        max_pitch = 0.5  # rad
        pitch_angle = self._model_data.qpos[4]
        if -max_pitch < pitch_angle < max_pitch:
            reward = 1.0
        else:
            reward = -1.0
        return reward

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
