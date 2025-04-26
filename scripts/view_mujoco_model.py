from importlib import resources

import mujoco.viewer

# Load the XML model
model_path = resources.files("balance") / "segway.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Create the viewer object
viewer = mujoco.viewer.launch_passive(model, data)


# Simulation loop
while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()

# Close the viewer
viewer.close()
