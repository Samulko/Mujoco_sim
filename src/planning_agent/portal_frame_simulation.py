import mujoco
import mujoco.viewer
import time
import os
import xml.etree.ElementTree as ET
import shutil
import threading

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

def simulate(model, data, duration):
    with mujoco.viewer.launch(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -20
        
        start_time = time.time()
        while time.time() - start_time < duration and viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time_to_sleep = max(0, 0.001 - (time.time() - step_start))
            time.sleep(time_to_sleep)

def run_simulation():
    try:
        while True:
            model, data = load_model(WORKING_XML_PATH)
            print("\nCurrent model information:")
            print_model_info(model)

            # Start the simulation in a separate thread
            sim_thread = threading.Thread(target=simulate, args=(model, data, 10))
            sim_thread.start()

            user_input = input("\nEnter element to remove (column1, column2, beam) or 'q' to quit: ")
            
            if user_input.lower() == 'q':
                break
            
            if user_input in ["column1", "column2", "beam"]:
                if remove_element(WORKING_XML_PATH, user_input):
                    print(f"Element {user_input} removed. Restarting simulation...")
                    model, data = load_model(WORKING_XML_PATH)
                else:
                    print("Failed to remove element. Continuing with current model.")
            else:
                print("Invalid input. Please try again.")

            # Wait for the simulation thread to finish
            sim_thread.join()

    except KeyboardInterrupt:
        print("\nExiting simulation...")
    finally:
        reset_xml()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_simulation()
