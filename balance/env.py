import math  # Import math for trigonometric functions
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from balance.checks import check_argument, check_state


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
    max_balanced_pitch: float = 0.5  # Max pitch to be considered balanced upright.
    max_standing_up_pitch: float = (
        1.8
        # Pitches above this will mean the robot has lost balance and laying down.
    )
    lay_down_grace_period: float = (
        1.0  # seconds to wait for a laying down robot to recover.
    )
    max_episode_duration: float = 10.0  # Max episode duration in seconds


@dataclass
class ResetSettings:
    max_speed: float = (
        0.5  # initial ground speed in m/s (within -max_speed and max_speed)
    )


class SegwayEnv(gym.Env):
    """
    A custom Gymnasium environment for the MuJoCo Segway model.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        sim_settings: SimulationSettings,
        behavior_settings: BehaviorSettings,
        reset_settings: ResetSettings,
    ):
        super().__init__()
        self._sim_settings = sim_settings
        self._model = model
        self._model_data = mujoco.MjData(self._model)
        self._sim_settings = sim_settings
        self._behavior = behavior_settings
        self._reset_settings = reset_settings

        # Set simulation options (can override XML)
        self._model.opt.timestep = self._sim_settings.sim_timestep

        self._initialize_part_ids()

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

        # --- Internal State for Truncation ---
        self.episode_steps = 0

        # --- Internal State for Termination ---
        self._lay_down_start_time = None
        self._episode_start_time = None

    def _initialize_part_ids(self):
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

        self._left_wheel_geom = self._model.geom("left-wheel-geom")
        self._left_wheel_body = self._model.body("left-wheel")

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
        self._episode_start_time = self._model_data.time

        info = {}
        return observation, info

    def _get_initial_velocity(self):
        # Set qvel (velocities) to a random value in the direction of the wheels
        max_speed = self._reset_settings.max_speed
        initial_speed = self.np_random.uniform(-max_speed, max_speed)
        yaw_angle = self._model_data.qpos[
            6
        ]  # Yaw angle is the last element of the orientation quaternion
        # Calculate the x and y components of the velocity based on the yaw angle
        vel_x = initial_speed * math.cos(yaw_angle)
        vel_y = initial_speed * math.sin(yaw_angle)
        vel = vel_x, vel_y, 0.0
        return vel

    def get_initial_pose(self):
        """
        Calculates the initial pose of the robot, including position and orientation.

        The height is adjusted based on the initial pitch angle to ensure the robot's
        wheels are on the ground, even when starting with a non-zero pitch.

        Returns:
            np.ndarray: The initial pose (qpos) of the robot."""
        pitch_angle = self.np_random.uniform(
            -self._sim_settings.max_init_pitch,
            self._sim_settings.max_init_pitch,
        )
        orientation = self.get_initial_orientation(pitch_angle)
        qpos = np.zeros(self._model.nq)
        qpos[0] = 0.0  # Initial x position
        qpos[1] = 0.0  # Initial y position
        qpos[2] = self._get_initial_height(pitch_angle)
        qpos[3:7] = orientation
        return qpos

    def _get_initial_height(self, pitch_angle):
        """
        Calculates the initial height adjustment based on the pitch angle.

        Args:
            pitch_angle (float): The initial pitch angle in radians.

        Returns:
            float: The adjusted height."""
        pass
        wheel_radius = self._left_wheel_geom.size[0]
        wheel_offset = abs(self._left_wheel_body.pos[2])
        height = wheel_radius + wheel_offset * math.cos(pitch_angle)
        print(
            f"Wheel radius: {wheel_radius}\nWheel offset: {wheel_offset}\nPitch: {pitch_angle}\n"
            f"Height: {height}"
        )
        return height

    def get_initial_orientation(self, pitch_angle: float):
        yaw_angle = self.np_random.uniform(-math.pi, math.pi)

        # noinspection PyArgumentList
        rotation = R.from_euler(
            "ZXY", [yaw_angle, pitch_angle, 0.0], degrees=False
        )  # Use False since angles are in radians

        orientation = rotation.as_quat(scalar_first=True)

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
        truncated = self._get_truncated_condition()

        # Example termination: Check if fallen over
        info = {}
        return observation, reward, terminated, truncated, info

    def _get_truncated_condition(self):
        elapsed = self._model_data.time - self._episode_start_time

        return elapsed > self._behavior.max_episode_duration

    def _get_terminated_condition(self):
        """
        Checks if the episode should be terminated based on the robot's state.
        """
        current_time = self._model_data.time
        pitch_angle = self._model_data.qpos[4]

        # Check if the robot is laying down
        if abs(pitch_angle) > self._behavior.max_standing_up_pitch:
            if self._lay_down_start_time is None:
                self._lay_down_start_time = current_time
            else:
                # Check if the robot has been laying down for too long
                if (
                    current_time - self._lay_down_start_time
                    > self._behavior.lay_down_grace_period
                ):
                    return True
        else:
            # Reset the lay down timer if the robot is upright
            self._lay_down_start_time = None

        return False

    def _get_reward(self):
        pitch_angle = self._model_data.qpos[4]
        if abs(pitch_angle) < self._behavior.max_balanced_pitch:
            reward = 1.0
        else:
            reward = -1.0
        return reward

    def close(self):
        """Closes the viewer if it's open."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
