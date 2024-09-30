import mujoco
import mujoco.viewer
import time
import os
import hashlib

def get_file_hash(filename):
    with open(filename, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def run_simulation():
    xml_path = "portal_frame.xml"
    last_modified_hash = get_file_hash(xml_path)

    def load_model():
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        model.opt.gravity[2] = -9.81
        model.opt.timestep = 0.002
        model.opt.integrator = 0
        return model, data

    model, data = load_model()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while True:
            current_hash = get_file_hash(xml_path)
            if current_hash != last_modified_hash:
                print("XML file changed. Reloading model...")
                model, data = load_model()
                viewer.load_model(model)
                last_modified_hash = current_hash

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

            if not viewer.is_running():
                break

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_simulation()
