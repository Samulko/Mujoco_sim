import mujoco
import mujoco.viewer
import time
import os

def run_simulation():
    # Load the model from the XML file
    xml_path = "portal_frame.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Set more realistic simulation parameters
    model.opt.gravity[2] = -9.81  # Ensure gravity is set correctly
    model.opt.timestep = 0.002    # Smaller timestep for more accurate simulation
    model.opt.integrator = 0      # Use semi-implicit Euler integrator for stability

    # Create a viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Run the simulation indefinitely
        while True:
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Update the viewer
            viewer.sync()
            
            # Add a small delay to control the simulation speed
            time.sleep(0.01)

            # Check if the viewer has been closed
            if not viewer.is_running():
                break

if __name__ == "__main__":
    # Change the working directory to the script's location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_simulation()
