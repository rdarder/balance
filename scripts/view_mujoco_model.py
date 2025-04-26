from importlib import resources
import time
import mujoco.viewer

# Load the XML model
model_path = resources.files("balance") / "segway.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# --- Set initial torque ---
# Find actuator indices (optional but good practice)
left_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left-motor")
right_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right-motor")

# Define the desired initial control signal corresponding to your desired torque
# Since Torque = gear * ctrl, then ctrl = Desired_Torque / gear
desired_initial_torque = 0.01  # Example: 0.001 Nm
gear = 0.0096  # Your gear value
initial_ctrl_signal = desired_initial_torque / gear

# Apply to the control array
if left_motor_id != -1:
    data.ctrl[left_motor_id] = initial_ctrl_signal
if right_motor_id != -1:
    data.ctrl[right_motor_id] = initial_ctrl_signal
# Or if you know they are the first two actuators:
# data.ctrl[0] = initial_ctrl_signal
# data.ctrl[1] = initial_ctrl_signal

# --- Launch viewer or run simulation loop ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)  # Apply the torque set in data.ctrl
        viewer.sync()
        # ... rest of your loop ...

        # Optional: Keep applying the torque every step if needed
        # data.ctrl[left_motor_id] = initial_ctrl_signal
        # data.ctrl[right_motor_id] = initial_ctrl_signal
