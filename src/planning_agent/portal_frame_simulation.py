import mujoco
import mujoco.viewer
import time
import os
import xml.etree.ElementTree as ET
import shutil
import threading
import numpy as np

STRUCTURE_COLLAPSED = False
initial_positions = {}

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

def initialize_positions(model, data):
    global initial_positions
    initial_positions = {i: data.qpos[7*i:7*i+3].copy() for i in range(1, model.nbody)}

def is_structure_collapsed(model, data, displacement_threshold=0.5, rotation_threshold=0.5, floor_threshold=0.1):
    for i in range(1, model.nbody):
        # Check total displacement
        current_pos = data.qpos[7*i:7*i+3]
        initial_pos = initial_positions[i]
        displacement = np.linalg.norm(current_pos - initial_pos)
        
        # Check rotation (simplified, using quaternion)
        quat = data.qpos[7*i+3:7*i+7]
        rotation_angle = 2 * np.arccos(np.abs(quat[0]))  # Angle from identity rotation
        
        # Check if body is close to the floor
        height_above_floor = current_pos[2]  # z-coordinate
        
        print(f"Debug: Body {i} - Displacement: {displacement:.4f}, Rotation: {rotation_angle:.4f}, Height: {height_above_floor:.4f}")
        
        if displacement > displacement_threshold or rotation_angle > rotation_threshold or height_above_floor < floor_threshold:
            return True
    return False

def reset_xml():
    shutil.copy2(ORIGINAL_XML_PATH, WORKING_XML_PATH)
    print("Reset XML file to original state")

def run_simulation():
    global STRUCTURE_COLLAPSED
    viewer = None
    viewer_thread = None

    def run_viewer(model, data, stop_event):
        nonlocal viewer
        try:
            viewer = mujoco.viewer.launch(model, data)
            if viewer is None:
                print("Failed to launch MuJoCo viewer. Simulation will run without visualization.")
                return
        
            def align_view():
                viewer.cam.lookat[:] = model.stat.center
                viewer.cam.distance = 10 * model.stat.extent
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20

            align_view()  # Initial alignment
        
            while not stop_event.is_set() and viewer.is_running():
                viewer.sync()
                time.sleep(0.01)
        except Exception as e:
            print(f"An error occurred in the viewer thread: {e}")
        finally:
            if viewer:
                viewer.close()

    try:
        while True:
            STRUCTURE_COLLAPSED = False
            model, data = load_model(WORKING_XML_PATH)
            print("\nCurrent model information:")
            print_model_info(model)

            stop_event = threading.Event()
            viewer_thread = threading.Thread(target=run_viewer, args=(model, data, stop_event))
            viewer_thread.start()

            user_input = input("\nEnter element to remove (column1, column2, beam) or 'q' to quit: ")

            if user_input.lower() == 'q':
                break

            if user_input in ["column1", "column2", "beam"]:
                if remove_element(WORKING_XML_PATH, user_input):
                    print(f"Element {user_input} removed. Restarting simulation...")
                    
                    # Reload the model after removing the element
                    model, data = load_model(WORKING_XML_PATH)
                    initialize_positions(model, data)  # Reset initial positions
                    
                    # Run the simulation for a few seconds to allow collapse to occur
                    for _ in range(500):  # Assuming 100 steps per second, this runs for 5 seconds
                        mujoco.mj_step(model, data)
                        if is_structure_collapsed(model, data):
                            STRUCTURE_COLLAPSED = True
                            print("Log: collapsed: true")
                            break
                        time.sleep(0.01)
                    
                    if not STRUCTURE_COLLAPSED:
                        print("Log: collapsed: false")
                else:
                    print("Failed to remove element. Continuing with current model.")
            else:
                print("Invalid input. Please try again.")

            print(f"Debug: STRUCTURE_COLLAPSED = {STRUCTURE_COLLAPSED}")

            # Close the viewer and stop the thread
            stop_event.set()
            if viewer_thread:
                viewer_thread.join(timeout=2)
            
            # Ensure the viewer is closed
            if viewer:
                viewer.close()
                viewer = None

    except KeyboardInterrupt:
        print("\nExiting simulation...")
    finally:
        if 'stop_event' in locals():
            stop_event.set()
        if viewer_thread and viewer_thread.is_alive():
            viewer_thread.join(timeout=2)
        if viewer:
            viewer.close()
        reset_xml()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_simulation()

import atexit
import signal
atexit.register(lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
