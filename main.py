# Modified main.py
# This script sets up a PyBullet simulation environment with the robot, tables, trays, and boxes.
# Debug parameters are enabled for manual control of the robot's joint angles and gripper opening.
# The simulation loop reads debug parameters and steps the environment accordingly using joint control.

import os
import numpy as np
import pybullet as p
import time
import math

# Removed tqdm as it was commented out in the original and not used in the provided env
# from tqdm import tqdm

# Import the custom environment and robot classes
from env import PickPlaceEnv
from robot import UR5Robotiq85 # Using the UR5 with Robotiq 85 gripper
# Removed Panda, UR5Robotiq140 as they are not used in this setup
# Removed Camera and YCBModels as they are not used in this version


def run_simulation():
    """
    Sets up the PyBullet simulation environment and runs simulation steps
    controlled by debug parameters for manual robot manipulation (joint control).
    """

    # Robot position: Centered between the two tables/trays (at z=table_height)
    # Table height is approximately 0.63m based on the table URDF.
    # Place the robot base slightly above the ground plane.
    robot_base_pos = [0, 0, 0.63]
    # Robot base orientation: [roll, pitch, yaw]. [0, 0, 0] points along positive X.
    robot_base_ori = [0, 0, 0]
    # Instantiate the specific robot model
    robot = UR5Robotiq85(robot_base_pos, robot_base_ori)

    # Instantiate the pick and place environment
    # Pass the robot instance and visualization setting. Camera is not used.
    # Set vis=True to see the GUI.
    env = PickPlaceEnv(robot=robot, vis=True)

    # Reset the environment to load objects and set initial robot pose
    # The reset method in env.py will handle spawning the tables, trays, robot, and boxes.
    env.reset()

    print("Environment setup complete. Use debug sliders in PyBullet GUI to control the robot's joint angles and gripper.")
    print("Press Ctrl+C in the console to exit.")

    # --- Control Loop (using Debug Parameters for Joint Control) ---
    # This loop continuously reads the debug parameters and applies them as actions.
    while True:
        try:
            # Read the target joint angles and gripper opening from debug parameters
            # env.read_debug_parameter() now returns a list of joint angles + gripper value
            action = env.read_debug_parameter()
            # Step the environment with the action using 'joint' control method.
            # The step method in env.py will apply the action and run simulation steps.
            obs, reward, done, info = env.step(action, 'joint') # Use 'joint' control

            # In this version, reward, done, and info are placeholders from env.step.
            # You can optionally print basic observation data if needed for debugging.
            # print(f"Current Joint Positions: {obs['positions']}")
            # print(f"Current EE Position: {obs['ee_pos']}")


            # The done condition is a placeholder in env.py, so the loop runs indefinitely
            # until interrupted. If you implement a real done condition, you would reset here.
            # if done:
            #     print("Episode finished. Resetting environment.")
            #     obs = env.reset()
            #     time.sleep(1) # Optional: pause before starting the next episode

        # Handle keyboard interrupt (Ctrl+C) to exit the simulation gracefully
        except KeyboardInterrupt:
            print("\nExiting simulation.")
            break
        # Catch any other exceptions during the simulation loop
        except Exception as e:
            print(f"An error occurred during simulation: {e}")
            # Optionally, break the loop or handle the error differently
            break

    # Clean up and disconnect from the PyBullet physics server
    env.close()


if __name__ == '__main__':
    # Run the main simulation function when the script is executed
    run_simulation()
