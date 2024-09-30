import mujoco
import mujoco.viewer
import time
import os
import xml.etree.ElementTree as ET
import shutil

ORIGINAL_XML_PATH = "portal_frame_original.xml"
WORKING_XML_PATH = "portal_frame.xml"

def load_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data

def remove_element(xml_path, element_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    for element in worldbody.findall('body'):
        if element.get('name') == element_name:
            worldbody.remove(element)
            tree.write(xml_path)
            print(f"Removed element: {element_name}")
            return True
    
    print(f"Element {element_name} not found")
    return False

def print_model_info(model):
    print(f"Number of bodies: {model.nbody}")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"Body {i}: {body_name}")

def reset_xml():
    shutil.copy2(ORIGINAL_XML_PATH, WORKING_XML_PATH)
    print("Reset XML file to original state")

def run_simulation():
    model, data = load_model(WORKING_XML_PATH)
    print("Initial model information:")
    print_model_info(model)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.distance = 5.0
            viewer.cam.elevation = -20

            last_input_time = time.time()
            while viewer.is_running():
                step_start = time.time()
                
                mujoco.mj_step(model, data)
                viewer.sync()

                current_time = time.time()
                if current_time - last_input_time >= 1.0:
                    user_input = input("Enter element to remove (column1, column2, beam) or press Enter to continue: ")
                    if user_input in ["column1", "column2", "beam"]:
                        if remove_element(WORKING_XML_PATH, user_input):
                            model, data = load_model(WORKING_XML_PATH)
                            viewer.close()
                            viewer = mujoco.viewer.launch_passive(model, data)
                            viewer.cam.azimuth = 90
                            viewer.cam.distance = 5.0
                            viewer.cam.elevation = -20
                            print("Model information after removal:")
                            print_model_info(model)
                    last_input_time = current_time

                time_to_sleep = max(0, 0.01 - (time.time() - step_start))
                time.sleep(time_to_sleep)
    except KeyboardInterrupt:
        print("\nExiting simulation...")
    finally:
        reset_xml()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_simulation()
