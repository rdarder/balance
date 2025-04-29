import time
import mujoco.viewer

from balance.utils import load_robot_model

# Load the XML model
model = load_robot_model()
model_data = mujoco.MjData(model)


# --- Set initial torque ---
# Find actuator indices (optional but good practice)
left_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left-motor")
right_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right-motor")

# very rough script that makes the segway stand up and land gracefully.
# just for vibe checking that the friction parameters and sim timing are about right.

initial_ctrl_signal = 1.0
reverse_ctrl_signal = -0.5
# --- Launch viewer or run simulation loop ---
with mujoco.viewer.launch_passive(model, model_data) as viewer:
    started = time.time()
    while viewer.is_running():
        step_start = time.time()
        elapsed = step_start - started
        if elapsed > 2.5:
            model_data.ctrl[left_motor_id] = 0
            model_data.ctrl[right_motor_id] = 0
        elif elapsed > 2.45:
            model_data.ctrl[left_motor_id] = reverse_ctrl_signal
            model_data.ctrl[right_motor_id] = reverse_ctrl_signal
        elif elapsed > 2.0:
            model_data.ctrl[left_motor_id] = initial_ctrl_signal
            model_data.ctrl[right_motor_id] = initial_ctrl_signal

        mujoco.mj_step(model, model_data)  # Apply the torque set in data.ctrl

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        viewer.sync()
        # ... rest of your loop ...

        # Optional: Keep applying the torque every step if needed
        # data.ctrl[left_motor_id] = initial_ctrl_signal
        # data.ctrl[right_motor_id] = initial_ctrl_signal
