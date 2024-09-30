import mujoco
import mujoco.viewer

# Define the MuJoCo model XML
xml = """
<mujoco>
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <geom name="floor" pos="0 0 0" size="5 5 0.1" type="plane" rgba="0.8 0.9 0.8 1"/>
        <body name="column1" pos="-1 0 1">
            <joint type="fixed"/>
            <geom name="column1" type="box" size="0.1 0.1 1" rgba="0.8 0.6 0.4 1"/>
        </body>
        <body name="beam" pos="0 0 2">
            <joint type="fixed"/>
            <geom name="beam" type="box" size="1.1 0.1 0.1" rgba="0.8 0.6 0.4 1"/>
        </body>
        <body name="column2" pos="1 0 1">
            <joint type="fixed"/>
            <geom name="column2" type="box" size="0.1 0.1 1" rgba="0.8 0.6 0.4 1"/>
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
            beam_z = data.body('beam').xpos[2]
            if beam_z < 1.8:  # Check if beam has fallen below 90% of its original height
                print(f"Structure collapsed {'without ' + remove_element if remove_element else ''}")
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
    for element in ["column1", "beam", "column2"]:
        run_simulation(model, remove_element=element)

if __name__ == "__main__":
    main()
