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

def simulate(model, data, viewer_created_event, stop_event):
    try:
        viewer = mujoco.viewer.launch(model, data)
        if viewer is None:
            print("Failed to launch MuJoCo viewer. Simulation will run without visualization.")
            return

        def align_view():
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance = 1.5 * model.stat.extent
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20

        align_view()  # Initial alignment
        viewer_created_event.set()  # Signal that the viewer has been created
        
        while viewer.is_running() and not stop_event.is_set():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time_to_sleep = max(0, 0.001 - (time.time() - step_start))
            time.sleep(time_to_sleep)
    except Exception as e:
        print(f"An error occurred during simulation: {e}")
    finally:
        if 'viewer' in locals() and viewer is not None:
            viewer.close()

def run_simulation():
    try:
        while True:
            model, data = load_model(WORKING_XML_PATH)
            print("\nCurrent model information:")
            print_model_info(model)

            stop_event = threading.Event()
            viewer_created_event = threading.Event()
            sim_thread = threading.Thread(target=simulate, args=(model, data, viewer_created_event, stop_event))
            sim_thread.start()

            # Wait for the viewer to be created before accepting input
            viewer_created_event.wait()
            print("Viewer created, waiting for 2 seconds before prompting for input...")
            time.sleep(2)  # Give some time for the viewer to stabilize

            print("Prompting for user input...")
            try:
                user_input = input("\nEnter element to remove (column1, column2, beam) or 'q' to quit: ")
                print(f"Received user input: {user_input}")
            except EOFError:
                print("\nEOF detected. Exiting simulation...")
                break

            stop_event.set()
            print("Stopping simulation thread...")
            sim_thread.join(timeout=5)  # Wait for up to 5 seconds
            if sim_thread.is_alive():
                print("Warning: Simulation thread did not stop in time.")
            else:
                print("Simulation thread stopped.")

            if user_input.lower() == 'q':
                break

            if user_input in ["column1", "column2", "beam"]:
                if remove_element(WORKING_XML_PATH, user_input):
                    print(f"Element {user_input} removed. Restarting simulation...")
                else:
                    print("Failed to remove element. Continuing with current model.")
            else:
                print("Invalid input. Please try again.")

            # Reload the model after removal attempt
            model, data = load_model(WORKING_XML_PATH)

    except KeyboardInterrupt:
        print("\nExiting simulation...")
    finally:
        reset_xml()
        print("Simulation ended.")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_simulation()
