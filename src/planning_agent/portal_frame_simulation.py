import numpy as np
import sys
import os
import platform
import traceback

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
    print("matplotlib is available. Visualization will be enabled.")
except ImportError:
    print("matplotlib is not installed. Visualization will be disabled.")
    print("To enable visualization, please install matplotlib using:")
    print("pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("Operating System:", platform.system(), platform.release())
print("CPU Architecture:", platform.machine())

try:
    import mujoco
    print("MuJoCo imported successfully")
    # Add a simple MuJoCo operation here
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    print("MuJoCo model created successfully")
except Exception as e:
    print(f"Error: {str(e)}")
    print("Traceback:")
    traceback.print_exc()
    sys.exit(1)

print("Imports successful")

# Test MuJoCo functionality
try:
    test_model = mujoco.MjModel.from_xml_string('<mujoco/>')
    test_data = mujoco.MjData(test_model)
    mujoco.mj_step(test_model, test_data)
    print("Basic MuJoCo functionality test passed")
except Exception as e:
    print("Error during basic MuJoCo functionality test:", str(e))
    print("Error details:")
    traceback.print_exc()
    sys.exit(1)

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

print("XML defined")

def run_simulation(model, remove_element=None, steps=1000):
    print(f"Starting simulation {'without ' + remove_element if remove_element else 'with all elements'}")
    if remove_element:
        try:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, remove_element)
            model.geom_rgba[geom_id] = [0, 0, 0, 0]
        except Exception as e:
            print(f"Error removing element {remove_element}: {e}")
            return None

    data = mujoco.MjData(model)

    try:
        for _ in range(steps):
            mujoco.mj_step(model, data)
            
            # Check if structure has fallen (you may need to adjust this threshold)
            fallen = any(data.qpos[2::7] < 0.5)  # Check z-position of bodies
            if fallen:
                print(f"Structure fell after {_} steps")
                return True

        print(f"Simulation completed {steps} steps without falling")
        return False
    except Exception as e:
        print(f"Error during simulation: {e}")
        return None

# Create the model
try:
    model = mujoco.MjModel.from_xml_string(xml)
    print("Model created successfully")
except Exception as e:
    print(f"Error creating model: {e}")
    print("MuJoCo error type:", type(e).__name__)
    print("MuJoCo error details:", str(e))
    sys.exit(1)

# Test basic MuJoCo operations
try:
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    print("Basic MuJoCo operations successful")
except Exception as e:
    print(f"Error during basic MuJoCo operations: {e}")
    print("MuJoCo error type:", type(e).__name__)
    print("MuJoCo error details:", str(e))
    sys.exit(1)

# Run simulations
print("Running simulation with all elements...")
fallen_complete = run_simulation(model)
if fallen_complete is not None:
    print(f"Structure {'has fallen' if fallen_complete else 'remains standing'}")
else:
    print("Simulation failed")

elements = ["column1", "beam", "column3"]
for element in elements:
    print(f"\nRunning simulation without {element}...")
    fallen = run_simulation(model, remove_element=element)
    if fallen is not None:
        print(f"Structure {'has fallen' if fallen else 'remains standing'}")
    else:
        print("Simulation failed")

print("Simulation complete")

def visualize_portal_frame(model, data):
    if not MATPLOTLIB_AVAILABLE:
        print("Visualization skipped: matplotlib is not available.")
        return

    plt.ion()  # Turn on interactive mode
    plt.clf()  # Clear the current figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot floor
    floor_pos = model.geom_pos[0]
    floor_size = model.geom_size[0]
    ax.plot_surface(
        [floor_pos[0]-floor_size[0], floor_pos[0]+floor_size[0]],
        [floor_pos[1]-floor_size[1], floor_pos[1]+floor_size[1]],
        [floor_pos[2], floor_pos[2]],
        alpha=0.5
    )
    
    # Plot columns and beam
    for i in range(1, 4):  # Assuming 3 bodies: 2 columns and 1 beam
        pos = data.xpos[i]
        if i == 2:  # Beam
            ax.plot([pos[0]-1, pos[0]+1], [pos[1], pos[1]], [pos[2], pos[2]], 'r-', linewidth=5)
        else:  # Columns
            ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [pos[2]-1, pos[2]+1], 'b-', linewidth=5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Portal Frame Visualization')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 3)
    plt.draw()
    plt.pause(0.001)  # Pause to update the plot

# Modify the run_simulation function to visualize the frame
def run_simulation(model, remove_element=None, steps=1000):
    print(f"Starting simulation {'without ' + remove_element if remove_element else 'with all elements'}")
    if remove_element:
        try:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, remove_element)
            model.geom_rgba[geom_id] = [0, 0, 0, 0]
        except Exception as e:
            print(f"Error removing element {remove_element}: {e}")
            return None

    data = mujoco.MjData(model)

    try:
        for step in range(steps):
            mujoco.mj_step(model, data)
            
            # Visualize every 100 steps
            if step % 100 == 0 and MATPLOTLIB_AVAILABLE:
                visualize_portal_frame(model, data)
            
            # Check if structure has fallen (you may need to adjust this threshold)
            fallen = any(data.qpos[2::7] < 0.5)  # Check z-position of bodies
            if fallen:
                print(f"Structure fell after {step} steps")
                if MATPLOTLIB_AVAILABLE:
                    visualize_portal_frame(model, data)  # Visualize the fallen state
                    plt.show(block=True)  # Keep the plot open
                return True

        print(f"Simulation completed {steps} steps without falling")
        if MATPLOTLIB_AVAILABLE:
            visualize_portal_frame(model, data)  # Visualize the final state
            plt.show(block=True)  # Keep the plot open
        return False
    except Exception as e:
        print(f"Error during simulation: {e}")
        return None
