from importlib import resources

import mujoco


def load_robot_model():
    global model, data
    model_path = resources.files("balance") / "segway.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return model

