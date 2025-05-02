import math
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- TF-Agents Imports ---
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
# -------------------------

from balance.checks import check_argument, check_state


@dataclass
class SimulationSettings:
    robot_hz: float = 50.0
    simulation_hz: float = 500.0
    max_init_axle_rotation: float = 1.8

    def __post_init__(self):
        check_argument(
            self.simulation_hz >= self.robot_hz
            or math.isclose(self.simulation_hz, self.robot_hz),
            "Simulation frequency must be greater than or equal to robot frequency",
        )
        check_argument(
            math.isclose(self.simulation_hz % self.robot_hz, 0.0)
            or math.isclose(self.simulation_hz % self.robot_hz, self.robot_hz),
            "Simulation frequency must be an integer multiple of robot frequency",
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
        return max(1, skip)


@dataclass
class BehaviorSettings:
    max_balanced_roll: float = (
        0.5  # Max axle rotation (rad) to be considered balanced upright.
    )
    max_standing_up_roll: float = (
        1.8
        # Axle rotations above this will mean the robot has lost balance and laying down.
    )
    lay_down_grace_period: float = (
        1.0  # seconds to wait for a laying down robot to recover.
    )
    max_episode_duration: float = 5.0  # Max episode duration in seconds


@dataclass
class ResetSettings:
    max_speed: float = (
        0.5  # initial ground speed in m/s (within -max_speed and max_speed)
    )


class SegwayEnv(py_environment.PyEnvironment):
    """
    A custom TF-Agents PyEnvironment for the MuJoCo Segway model.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        model_data: mujoco.MjData,
        sim_settings: SimulationSettings,
        behavior_settings: BehaviorSettings,
        reset_settings: ResetSettings,
    ):

        self._sim_settings = sim_settings
        self._model = model
        self._model_data = model_data
        self._behavior = behavior_settings
        self._reset_settings = reset_settings

        # Set simulation options (can override XML)
        self._model.opt.timestep = self._sim_settings.sim_timestep

        self._initialize_part_ids()

        # --- Define Specs ---
        # Action spec: [left_motor_pwm_duty_cycle, right_motor_pwm_duty_cycle]
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0, name='action'
        )

        # Observation spec: [imu_accel (3), imu_gyro (3), desired_speed (1), desired_turn (1)]
        obs_dim = 6 + 2
        # Using more specific bounds per component is better practice long-term
        max_obs_val = 2 * math.pi # Placeholder bound
        min_obs_val = -max_obs_val
        # Ensure bounds are arrays matching shape
        bounds = np.array([20, 20, 20, 4, 4, 4, 1, 1], dtype=np.float32)

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(obs_dim,), dtype=np.float32, minimum=-bounds, maximum=bounds,
            name='observation'
        )
        # --------------------

        # --- Internal State for Desired Commands ---
        self._desired_speed = 0.0
        self._desired_turn = 0.0

        # --- Viewer (Optional) ---
        self.viewer = None

        # --- Internal State for Truncation/Termination ---
        self._episode_steps = 0
        self._lay_down_start_time = None
        self._episode_start_time = None
        self._episode_ended = False # Flag to indicate episode end state

    # --- Implement PyEnvironment abstract methods ---

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


    # --- Optional PyEnvironment methods ---
    def get_info(self):
        """Return auxiliary information for the current time step."""
        # Can return diagnostic info similar to gym's info dict
        # For now, let's return the roll angle
        return {'axle_angle_rad': self._get_current_roll_rad()}

    def get_state(self):
        """Return the current state of the environment."""
        # A dict containing non-PyObject state.
        # For MuJoCo, saving/restoring model_data might be complex.
        # Start simple or raise NotImplementedError
        state = {
            'model_data_time': self._model_data.time,
            'model_data_qpos': self._model_data.qpos.copy(),
            'model_data_qvel': self._model_data.qvel.copy(),
            'model_data_act': self._model_data.act.copy(),
            'model_data_ctrl': self._model_data.ctrl.copy(),
            'desired_speed': self._desired_speed,
            'desired_turn': self._desired_turn,
            'episode_steps': self._episode_steps,
            'lay_down_start_time': self._lay_down_start_time,
            'episode_start_time': self._episode_start_time,
            '_episode_ended': self._episode_ended
        }
        return state


    def set_state(self, state):
        """Set the current state of the environment."""
        # Restore state from the dict returned by get_state.
        self._model_data.time = state['model_data_time']
        self._model_data.qpos[:] = state['model_data_qpos']
        self._model_data.qvel[:] = state['model_data_qvel']
        self._model_data.act[:] = state['model_data_act']
        self._model_data.ctrl[:] = state['model_data_ctrl']
        # It's crucial to re-compute derived MuJoCo quantities after setting state
        mujoco.mj_forward(self._model, self._model_data)

        self._desired_speed = state['desired_speed']
        self._desired_turn = state['desired_turn']
        self._episode_steps = state['episode_steps']
        self._lay_down_start_time = state['lay_down_start_time']
        self._episode_start_time = state['episode_start_time']
        self._episode_ended = state['_episode_ended']


    # --- Existing Helper Methods (Mostly Unchanged) ---

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
        self._imu_accel_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel"
        )
        check_state(self._imu_accel_id != -1, "IMU accel not found in model")
        self._imu_accel_offset = self._model.sensor_adr[self._imu_accel_id]
        self._imu_gyro_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
        )
        check_state(self._imu_gyro_id != -1, "IMU gyro not found in model")
        self._imu_gyro_offset = self._model.sensor_adr[self._imu_gyro_id]
        self._chassis_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
        )
        check_state(self._chassis_body_id != -1, "Chassis body not found in model")
        self._left_wheel_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "left-wheel-geom"
        )
        check_state(self._left_wheel_geom_id != -1, "Left wheel geom not found")
        self._left_wheel_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "left-wheel"
        )
        check_state(self._left_wheel_body_id != -1, "Left wheel body not found")

    def set_movement_commands(self, speed: float, turn: float):
        """Sets the desired speed and turn commands for the agent to follow."""
        self._desired_speed = np.clip(speed, -1.0, 1.0)
        self._desired_turn = np.clip(turn, -1.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        """Collects the current observation."""
        imu_accel = self._model_data.sensordata[
            self._imu_accel_offset :self._imu_accel_offset + 3
        ]
        imu_gyro = self._model_data.sensordata[
            self._imu_gyro_offset: self._imu_gyro_offset + 3
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

    def _reset(self):
        """Starts a new episode and returns the first `TimeStep`."""
        self._episode_ended = False

        # Reset the MuJoCo simulation data to initial XML state first
        mujoco.mj_resetData(self._model, self._model_data)

        # --- Set Initial Pose and Velocity ---
        initial_pose = self.get_initial_pose()
        self._model_data.qpos[:] = initial_pose
        initial_orientation_quat = initial_pose[3:7]
        self._model_data.qvel[0:3] = self._get_initial_velocity(initial_orientation_quat)
        self._model_data.qvel[3:] = 0.0
        # ------------------------------------

        # --- Crucial: Forward dynamics ---
        mujoco.mj_forward(self._model, self._model_data)
        # ---------------------------------

        # Reset desired commands (sample new random targets for the episode)
        self._desired_speed = np.random.uniform(-1, 1)
        self._desired_turn = np.random.uniform(-1, 1)

        # Get the initial observation based on the new state
        observation = self._get_obs()

        # Reset internal state
        self._episode_steps = 0
        self._episode_start_time = self._model_data.time
        self._lay_down_start_time = None

        # Return the first TimeStep
        return ts.restart(observation)

    def _get_initial_velocity(self, orientation_quat):
        """Calculates initial velocity based on random speed and current yaw."""
        max_speed = self._reset_settings.max_speed
        initial_speed = np.random.uniform(-max_speed, max_speed) # Use np.random
        quat_xyzw = [
            orientation_quat[1],
            orientation_quat[2],
            orientation_quat[3],
            orientation_quat[0],
        ]
        # noinspection PyArgumentList
        rotation = R.from_quat(quat_xyzw)
        euler_angles = rotation.as_euler("zyx", degrees=False)
        yaw_angle = euler_angles[0]
        vel_x = initial_speed * math.cos(yaw_angle)
        vel_y = initial_speed * math.sin(yaw_angle)
        vel = np.array([vel_x, vel_y, 0.0], dtype=np.float32)
        return vel

    def get_initial_pose(self):
        """Calculates the initial pose of the robot."""
        # (Content is the same, uses np.random now)
        axle_roll_angle = np.random.uniform( # Use np.random
            -self._sim_settings.max_init_axle_rotation,
            self._sim_settings.max_init_axle_rotation,
        )
        orientation = self.get_initial_orientation(axle_roll_angle)
        qpos = np.zeros(self._model.nq, dtype=np.float64) # qpos is float64
        qpos[0] = 0.0
        qpos[1] = 0.0
        qpos[2] = self._get_initial_height(axle_roll_angle)
        qpos[3:7] = orientation
        return qpos

    def _get_initial_height(self, axle_angle):
        """Calculates the initial height adjustment."""
        wheel_radius = self._model.geom_size[self._left_wheel_geom_id][0]
        wheel_offset_z = self._model.body_pos[self._left_wheel_body_id][2]
        height = wheel_radius - wheel_offset_z * math.cos(axle_angle)
        return height

    def get_initial_orientation(self, axle_angle: float):
        """Calculates orientation based on random yaw and specified axle rotation (roll)."""
        # (Content is the same, uses np.random now)
        yaw_angle = np.random.uniform(-math.pi, math.pi) # Use np.random
        pitch_angle = 0.0
        roll_angle = axle_angle
        euler_angles = [yaw_angle, pitch_angle, roll_angle]
        # noinspection PyArgumentList
        rotation = R.from_euler("ZYX", euler_angles, degrees=False)
        # Get quaternion in MuJoCo's [w, x, y, z] format
        orientation = rotation.as_quat(scalar_first=True)
        return orientation

    def _step(self, action):
        """Applies the action, advances the simulation, and returns a `TimeStep`."""

        if self._episode_ended:
            return self.reset()

        action = np.clip(action, self._action_spec.minimum, self._action_spec.maximum)
        self._model_data.ctrl[self._left_motor_id] = action[0]
        self._model_data.ctrl[self._right_motor_id] = action[1]

        for _ in range(self._sim_settings.sim_frames_to_robot_frames):
            mujoco.mj_step(self._model, self._model_data)

        self._episode_steps += 1

        observation = self._get_obs()
        reward = self._get_reward() # Ignore info dict from reward func for now
        # --------------------------------

        # --- Check for Termination or Truncation ---
        terminated = self._get_terminated_condition()
        truncated = self._get_truncated_condition()
        self._episode_ended = terminated or truncated
        # -----------------------------------------

        # --- Return TimeStep ---
        if self._episode_ended:
            # Use termination for both terminated and truncated cases
            # The discount factor is automatically set to 0.0 by ts.termination
            return ts.termination(observation, reward)
        else:
            # Use transition for ongoing steps
            # The discount factor should be 1.0 for non-terminal steps
            return ts.transition(observation, reward, discount=1.0)
        # -----------------------


    def _get_truncated_condition(self):
        """Checks if episode duration limit is reached."""
        elapsed = self._model_data.time - self._episode_start_time
        return elapsed > self._behavior.max_episode_duration

    def _get_current_roll_rad(self) -> float:
        """Calculates the current roll angle (rotation around local X-axis)."""
        q = self._model_data.xquat[self._chassis_body_id]
        # noinspection PyArgumentList
        rotation = R.from_quat(q, scalar_first=True)

        euler_angles = rotation.as_euler("zyx", degrees=False)
        return euler_angles[2]

    def _get_terminated_condition(self):
        """Checks if the robot has fallen over for too long."""
        current_time = self._model_data.time
        current_roll_rad = self._get_current_roll_rad()

        if abs(current_roll_rad) > self._behavior.max_standing_up_roll:
            if self._lay_down_start_time is None:
                self._lay_down_start_time = current_time
            else:
                if (
                    current_time - self._lay_down_start_time
                    > self._behavior.lay_down_grace_period
                ):
                    return True
        else:
            self._lay_down_start_time = None
        return False

    def _get_reward(self) -> np.float32:
        """Calculates reward based on axle rotation (roll)."""
        current_roll_rad = self._get_current_roll_rad()

        # Reward based on being balanced, not just standing up
        if abs(current_roll_rad) < self._behavior.max_balanced_roll:
            reward = 1.0
        else:
            reward = -1.0

        return np.float32(reward)

    def close(self):
        """Closes the viewer if it's open."""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception as e:
                print(f"Error closing viewer: {e}")
            finally:
                self.viewer = None
