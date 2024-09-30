import mujoco
import mujoco.viewer
import time
import os
import hashlib
import xml.etree.ElementTree as ET

def get_file_hash(filename):
    with open(filename, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data

def add_elements(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    elements = [
        ('element1', '-1 0 2', '0.1 0.1 1', '0.8 0.6 0.4 1'),
        ('element2', '1 0 2', '0.1 0.1 1', '0.8 0.6 0.4 1'),
        ('element3', '0 0 3.1', '1 0.1 0.1', '0.8 0.6 0.6 1')
    ]
    
    for name, pos, size, rgba in elements:
        body = ET.SubElement(worldbody, 'body', name=name, pos=pos)
        ET.SubElement(body, 'freejoint')
        ET.SubElement(body, 'geom', name=name, type='box', size=size, rgba=rgba)
    
    tree.write(xml_path)

def remove_element(xml_path, element_number):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    element_name = f"element{element_number}"
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

def run_simulation():
    xml_path = "portal_frame.xml"
    add_elements(xml_path)
    last_modified_hash = get_file_hash(xml_path)

    model, data = load_model(xml_path)
    print("Initial model information:")
    print_model_info(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.distance = 10.0
        viewer.cam.elevation = -20

        last_input_time = time.time()
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

            current_time = time.time()
            if current_time - last_input_time >= 1.0:
                user_input = input("Enter element number to remove (1, 2, 3) or press Enter to continue: ")
                if user_input in ["1", "2", "3"]:
                    if remove_element(xml_path, user_input):
                        model, data = load_model(xml_path)
                        viewer.close()
                        viewer = mujoco.viewer.launch_passive(model, data)
                        viewer.cam.azimuth = 90
                        viewer.cam.distance = 10.0
                        viewer.cam.elevation = -20
                        print("Model information after removal:")
                        print_model_info(model)
                last_input_time = current_time

            time.sleep(0.01)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_simulation()
