import mujoco
import mujoco.viewer
import time
import os
import hashlib

def get_file_hash(filename):
    with open(filename, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data

def run_simulation():
    xml_path = "portal_frame.xml"
    last_modified_hash = get_file_hash(xml_path)

    model, data = load_model(xml_path)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            current_hash = get_file_hash(xml_path)
            if current_hash != last_modified_hash:
                print("XML file changed. Reloading model...")
                model, data = load_model(xml_path)
                viewer.close()
                viewer = mujoco.viewer.launch_passive(model, data)
                last_modified_hash = current_hash

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_simulation()
