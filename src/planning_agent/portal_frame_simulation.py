import mujoco
import mujoco.viewer

# Define the MuJoCo model XML
xml = """
<mujoco>
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <geom name="floor" pos="0 0 0" size="5 5 0.1" type="plane" rgba="0.8 0.9 0.8 1"/>
        <body name="column1" pos="-1 0 1">
            <joint type="free"/>
            <geom name="column1" type="capsule" size="0.1 1" rgba="0.8 0.6 0.4 1"/>
        </body>
        <body name="beam" pos="0 0 2">
            <joint type="free"/>
            <geom name="beam" type="capsule" size="0.1 1" rgba="0.8 0.6 0.4 1" euler="0 1.57 0"/>
        </body>
        <body name="column3" pos="1 0 1">
            <joint type="free"/>
            <geom name="column3" type="capsule" size="0.1 1" rgba="0.8 0.6 0.4 1"/>
        </body>
    </worldbody>
</mujoco>
"""

def run_simulation(model, remove_element=None, steps=1000):
    if remove_element:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, remove_element)
        model.geom_rgba[geom_id] = [0, 0, 0, 0]  # Make the element invisible

    data = mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(steps):
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Check if structure has fallen
            if any(data.qpos[2::7] < 0.5):  # Check z-position of bodies
                print(f"Structure fell {'without ' + remove_element if remove_element else ''}")
                viewer.sync()
                input("Press Enter to continue...")
                return True

        print(f"Structure remains standing {'without ' + remove_element if remove_element else ''}")
        viewer.sync()
        input("Press Enter to continue...")
        return False

def main():
    model = mujoco.MjModel.from_xml_string(xml)
    
    # Run simulation with all elements
    run_simulation(model)
    
    # Run simulations removing each element
    for element in ["column1", "beam", "column3"]:
        run_simulation(model, remove_element=element)

if __name__ == "__main__":
    main()
