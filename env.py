# Modified env.py
# Defines the PyBullet environment for a pick and place task, with spawning and debug control.
# Includes loading of the robot, tables, trays, and objects.
# Debug parameters are set up for joint angle control and gripper opening.
# Table and object placement positions are adjusted. A support cuboid is added beneath the robot.
# Implemented the get_observation method to return a structured observation for RL,
# including placeholders for camera data. A camera is instantiated but not active by default.
# Added and implemented _calculate_reward and _check_done methods.
# Increased the scale of reward and penalty values and added more detailed reward shaping.

import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data # Contains standard PyBullet URDFs

# Assuming utilities.py contains Camera and potentially a general Models base class
# Camera is now integrated but optional for observation.
from utilities import Camera
from collections import namedtuple
from attrdict import AttrDict

# Custom exception for failure cases (not used in this version but kept)
class FailToReachTargetError(RuntimeError):
    pass


class PickPlaceEnv:
    """
    PyBullet environment for a pick and place task, with spawning and debug control.
    Implements a structured observation space, reward function, and done condition for reinforcement learning.
    """

    # Simulation step delay for visualization purposes (seconds)
    SIMULATION_STEP_DELAY = 1 / 240. # Corresponds to 240 Hz simulation frequency

    # Define properties and positions for environment elements
    # TABLE_1_BASE_POS: Base position of the first table [x, y, z]. Z is typically 0.
    # TABLE_2_BASE_POS: Base position of the second table [x, y, z]. Z is typically 0.
    # Positions are relative to the world origin. Adjusted to move tables further apart along the Y axis.
    TABLE_1_BASE_POS = [0, 0.75, 0]
    TABLE_2_BASE_POS = [0, -0.75, 0]

    # TRAY_1_POS: Position of the first tray on top of Table 1 [x, y, z].
    # TRAY_2_POS: Position of the second tray on top of Table 2 [x, y, z].
    # Standard table.urdf top surface is around z=0.625m when base is at z=0.
    # Placing the tray slightly above this (e.g., 0.65m).
    TABLE_TOP_Z = 0.625 # Approximate Z coordinate of the top of the table
    TRAY_1_POS = [TABLE_1_BASE_POS[0], TABLE_1_BASE_POS[1], TABLE_TOP_Z + 0.025] # Place tray slightly above table
    TRAY_2_POS = [TABLE_2_BASE_POS[0], TABLE_2_BASE_POS[1], TABLE_TOP_Z + 0.025] # Place tray slightly above table

    # Approximate dimensions of the tray for calculating object spawn bounds and target area
    # These dimensions are based on the collision box in the standard pybullet_data tray.urdf
    TRAY_DIMENSIONS = [0.6, 0.6, 0.02] # (x, y, z) dimensions of the base collision box
    # Side length of the box object. This size is used for programmatic creation.
    BOX_SIDE_LENGTH = 0.05 # 5 cm side length, should fit the Robotiq 85 gripper (max 8.5cm)
    BOX_MASS = 0.1 # Mass of each box in kg
    NUM_BOXES = 4 # Number of boxes to spawn

    # Support cuboid properties
    SUPPORT_CUBOID_HALF_EXTENTS = [0.15, 0.15, 0.05] # Half dimensions of the support cuboid
    # Position the support cuboid just below the robot base (robot base is at z=0.63)
    SUPPORT_CUBOID_BASE_POS = [0, 0, 0.63 - SUPPORT_CUBOID_HALF_EXTENTS[2]] # Center position

    # Camera properties
    CAMERA_POS = [0.5, 0, 1.2] # Camera position [x, y, z]
    CAMERA_TARGET_POS = [0, 0, 0.7] # Point camera towards the center of the workspace
    CAMERA_UP_VECTOR = [0, 0, 1] # Up direction for the camera
    CAMERA_NEAR = 0.1 # Near plane distance
    CAMERA_FAR = 2.0 # Far plane distance
    CAMERA_SIZE = (320, 320) # Image width and height
    CAMERA_FOV = 60 # Field of view in degrees

    # Reward function parameters (Scaled up)
    REWARD_GRASP_SUCCESS = 100.0 # Reward for successfully grasping a box (Scaled from 10.0)
    REWARD_BOX_IN_TARGET = 500.0 # Reward for placing a box in the target tray (Scaled from 50.0)
    REWARD_ALL_BOXES_IN_TARGET = 1000.0 # Additional reward for placing all boxes (Scaled from 100.0)
    REWARD_STEP_PENALTY = -0.1 # Small penalty per step to encourage efficiency (Scaled from -0.01)
    REWARD_DROP_PENALTY = -50.0 # Penalty for dropping a grasped box (optional, can be tricky) (Scaled from -5.0)
    REWARD_COLLISION_PENALTY = -10.0 # Penalty for collisions (optional) (Scaled from -1.0)

    # Added/Adjusted Reward Shaping Parameters
    REWARD_CLOSER_TO_OBJECT_SCALE = 2.0 # Scale for rewarding movement towards the object (smaller reward)
    REWARD_TRANSITION_GRASP = 200.0 # Huge reward for successfully transitioning to a grasped state
    REWARD_ABOVE_TABLE = 0.001 # Very minute reward for EE staying above the table
    PENALTY_BELOW_TABLE = -0.5 # Negative reward for EE going below the table

    # Done condition parameters
    MAX_ENVIRONMENT_STEPS = 1000 # Maximum number of environment steps per episode
    TARGET_POS_TOLERANCE = 0.1 # Tolerance for considering a box "in" the target tray (meters)


    def __init__(self, robot, vis=False) -> None: # Removed camera and num_boxes from __init__ args
        """
        Initializes the PickPlaceEnv for spawning and visualization with debug control.

        Args:
            robot: An instance of a RobotBase subclass (e.g., UR5Robotiq85).
            vis: Boolean, set to True to run the simulation with a GUI for visualization.
        """
        self.robot = robot
        self.vis = vis # Visualization flag
        if self.vis:
            # Simple step counter for printing progress
            self._step_counter = 0
            pass
        # Camera is instantiated here, but self.camera will be None unless explicitly created
        # in main.py or elsewhere and passed to the environment.
        # Keeping self.camera = None for now as requested.
        self.camera = None # Set to None initially

        # Connect to the PyBullet physics server
        # Use GUI mode if vis is True, otherwise use DIRECT (headless) mode
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        # Add the standard PyBullet data path to search for URDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Set the gravity vector
        p.setGravity(0, 0, -9.81) # Standard gravity

        # Load static environment elements
        self.planeID = p.loadURDF("plane.urdf") # Load the ground plane
        # Load tables at their base positions from pybullet_data.
        # The standard table.urdf has its origin at the base, and the top surface is at z=0.625.
        self.table1ID = p.loadURDF("table/table.urdf", basePosition=self.TABLE_1_BASE_POS, baseOrientation=p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)
        self.table2ID = p.loadURDF("table/table.urdf", basePosition=self.TABLE_2_BASE_POS, baseOrientation=p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)

        # Load trays on top of the tables at the specified TRAY_POS z-coordinate.
        # Loading tray.urdf from pybullet_data path. Tray positions are on top of the tables.
        self.tray1ID = p.loadURDF("tray/tray.urdf", basePosition=self.TRAY_1_POS, useFixedBase=True)
        self.tray2ID = p.loadURDF("tray/tray.urdf", basePosition=self.TRAY_2_POS, useFixedBase=True)

        # --- Add support cuboid beneath the robot ---
        support_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                      halfExtents=self.SUPPORT_CUBOID_HALF_EXTENTS,
                                                      rgbaColor=[0.8, 0.8, 0.8, 1.0]) # Grey color
        support_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                            halfExtents=self.SUPPORT_CUBOID_HALF_EXTENTS)
        # Corrected: Removed useFixedBase=True from createMultiBody call
        self.supportCuboidID = p.createMultiBody(baseMass=0, # Fixed base (mass 0)
                                                 baseCollisionShapeIndex=support_collision_shape_id,
                                                 baseVisualShapeIndex=support_visual_shape_id,
                                                 basePosition=self.SUPPORT_CUBOID_BASE_POS,
                                                 baseOrientation=p.getQuaternionFromEuler([0,0,0]))
        # --- End add support cuboid ---


        # List to store the IDs of the spawned objects (boxes)
        self.object_ids = []
        # List to track which boxes have been successfully placed in the target tray
        self._boxes_in_target = []
        # Flag to track if the robot is currently grasping a box
        self._is_grasping = False
        # Flag to track grasping state in the previous step for transition reward
        self._prev_is_grasping = False


        # Store the previous distance to the object for reward shaping
        self._prev_dist_to_object = None
        # Store the previous distance to the target for reward shaping
        self._prev_dist_to_target = None


        # Load the robot model. The initial position and orientation is set in the robot instance.
        # Robot base position is still centered between the tables.
        self.robot.load()
        # Hook the environment's step_simulation method into the robot,
        # so robot control commands trigger simulation steps.
        self.robot.step_simulation = self.step_simulation # Hook simulation step

        # --- Add debug parameters for joint control and gripper opening ---
        # Create sliders for each controllable joint angle
        self.joint_debug_params = []
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
             joint_info = self.robot.joints[joint_id]
             # Use joint name for the slider label
             param_id = p.addUserDebugParameter(f"{joint_info.name}",
                                                joint_info.lowerLimit,
                                                joint_info.upperLimit,
                                                self.robot.arm_rest_poses[i]) # Initial value is the rest pose
             self.joint_debug_params.append(param_id)

        # Gripper control slider. Range from closed (0) to open (max of robot's gripper range).
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, self.robot.gripper_range[1], self.robot.gripper_range[1])
        # --- End add debug parameters ---


        # Counter for environment steps (distinct from simulation steps)
        self._env_step_counter = 0


    def step_simulation(self):
        """
        Steps the PyBullet physics simulation.
        This method is hooked into the robot class for synchronized control.
        """
        p.stepSimulation()
        if self.vis:
            # Optional: add a delay for real-time visualization or update a counter/progress bar
            # time.sleep(self.SIMULATION_STEP_DELAY)
            self._step_counter += 1
            # Print simulation step progress periodically
            if self._step_counter % 240 == 0: # Print every second of simulation time
                 # print(f"Sim Step: {self._step_counter}") # Commented out to reduce console spam during training
                 pass
            pass

    def read_debug_parameter(self):
        """
        Reads the current values from the debug sliders in the PyBullet GUI.
        Returns the target joint angles and gripper opening length.
        """
        # Read values from joint angle sliders
        target_joint_poses = [p.readUserDebugParameter(param_id) for param_id in self.joint_debug_params]
        # Read value from gripper slider
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        # Return joint angles followed by gripper opening length
        return target_joint_poses + [gripper_opening_length]

    def step(self, action, control_method='joint'):
        """
        Applies an action to the robot and steps the environment.
        Calculates reward and checks for done condition.

        Args:
            action: A tuple or list representing the desired robot action.
                    Format depends on control_method:
                    - 'end': (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                    - 'joint': List of target joint positions for arm_controllable_joints followed by gripper_opening_length.
            control_method: String, specifies the type of control ('end' or 'joint').
                            For debug control, this is expected to be 'joint'.

        Returns:
            obs: Dictionary containing observations.
            reward: Float, the reward for this step.
            terminated: Boolean, True if the episode terminated (e.g., task completed or failed).
            truncated: Boolean, True if the episode was truncated (e.g., time limit reached).
            info: Dictionary, additional diagnostic information.
        """
        # Validate control method (expecting 'joint' for this debug setup)
        assert control_method in ('joint', 'end'), "Invalid control_method. Use 'joint' or 'end'."
        # In this debug setup, we primarily use 'joint' control.
        if control_method != 'joint':
             # print(f"Warning: Expected 'joint' control_method, but received '{control_method}'. Using 'joint'.") # Commented out to reduce spam
             control_method = 'joint'

        # Apply the action to the robot's arm (joint positions)
        self.robot.move_ee(action[:-1], control_method) # Pass joint angles
        # Apply the action to the robot's gripper
        self.robot.move_gripper(action[-1])

        # Step the simulation multiple times for each environment step
        num_sim_steps = 120 # Number of simulation steps per environment step (e.g., 0.5 seconds at 240Hz)
        for _ in range(num_sim_steps):
            self.step_simulation()

        # Update environment step counter
        self._env_step_counter += 1

        # Get the current observation
        obs = self.get_observation()

        # Calculate the reward for this step
        reward = self._calculate_reward()

        # Check if the episode is done (terminated or truncated)
        terminated, truncated = self._check_done()

        # Gather additional information (optional)
        info = {} # Can add information like success rate, number of boxes placed, etc.

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self):
        """
        Calculates the reward for the current environment state.
        Implements a shaped reward function to encourage picking and placing boxes.
        Includes additional shaping for reaching, grasping, and vertical position.
        """
        reward = self.REWARD_STEP_PENALTY # Apply a small penalty per step

        # Get current state information
        robot_ee_position = self.robot.get_joint_obs()['ee_pos']
        gripper_opening_length = self.get_observation()['gripper_opening_length'][0] # Get scalar value
        object_states_flat = self.get_observation()['object_states']
        object_poses = object_states_flat.reshape(-1, 6) # Reshape to get individual object poses

        # --- Reward Shaping Implementation ---

        # 1. Reward for getting closer to the nearest ungrasped object:
        # Find the nearest ungrasped object
        nearest_obj_pos = None
        min_dist_to_obj = float('inf')
        for i, obj_id in enumerate(self.object_ids):
            if obj_id not in self._boxes_in_target: # Only consider ungrasped, unplaced objects
                obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
                dist_to_obj = np.linalg.norm(np.array(robot_ee_position) - np.array(obj_pos))
                if dist_to_obj < min_dist_to_obj:
                    min_dist_to_obj = dist_to_obj
                    nearest_obj_pos = obj_pos

        if nearest_obj_pos is not None:
            if self._prev_dist_to_object is not None:
                # Reward change in distance (positive for decreasing distance)
                # Using a smaller scaling factor as requested for this reward.
                reward += (self._prev_dist_to_object - min_dist_to_obj) * self.REWARD_CLOSER_TO_OBJECT_SCALE
            self._prev_dist_to_object = min_dist_to_obj # Update previous distance

        # Check for grasped objects (same logic as before)
        min_gripper_opening_for_grasp = 0.01 # Approximate minimum opening for a grasp
        box_lifted_height_threshold = self.TRAY_1_POS[2] + self.BOX_SIDE_LENGTH # Slightly above tray height

        grasped_box_id = None
        for i, obj_id in enumerate(self.object_ids):
             obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
             if gripper_opening_length < min_gripper_opening_for_grasp and obj_pos[2] > box_lifted_height_threshold:
                  ee_obj_distance = np.linalg.norm(np.array(robot_ee_position) - np.array(obj_pos))
                  if ee_obj_distance < self.BOX_SIDE_LENGTH * 2: # If EE is reasonably close to the lifted object
                       grasped_box_id = obj_id
                       break

        # Update current grasping state
        current_is_grasping = grasped_box_id is not None

        # 2. Huge reward if it picks up (transition from not grasping to grasping):
        if current_is_grasping and not self._prev_is_grasping:
             reward += self.REWARD_TRANSITION_GRASP
             print("Grasp transition detected! Huge reward awarded.") # Optional: Print for debugging

        # Update previous grasping state
        self._prev_is_grasping = current_is_grasping
        self._is_grasping = current_is_grasping # Keep _is_grasping updated for other checks

        # 3. Very minute reward if the gripper stays above the table:
        # 4. Negative reward if it goes below the table:
        # Using the approximate table top height for this check.
        if robot_ee_position[2] > self.TABLE_TOP_Z:
             reward += self.REWARD_ABOVE_TABLE
        else:
             reward += self.PENALTY_BELOW_TABLE # Penalty for being below the table

        # Reward for moving a grasped box towards the target tray (existing logic)
        if self._is_grasping and grasped_box_id is not None:
             grasped_obj_pos, _ = p.getBasePositionAndOrientation(grasped_box_id)
             target_tray_center_xy = np.array(self.TRAY_2_POS[:2])
             obj_pos_xy = np.array(grasped_obj_pos[:2])
             distance_to_target_xy = np.linalg.norm(target_tray_center_xy - obj_pos_xy)

             # Reward for reducing distance to target (negative reward for distance)
             # You can adjust the scaling factor here.
             reward += -distance_to_target_xy * 10.0 # Scaled from 1.0 (Example: Increased scaling)

             # Reward for lifting the box (if not already high enough)
             # This encourages lifting after grasping
             if grasped_obj_pos[2] < self.TRAY_2_POS[2] + self.BOX_SIDE_LENGTH: # If box is below target tray height
                  reward += (grasped_obj_pos[2] - box_lifted_height_threshold) * 100.0 # Scaled from 50.0 (Example: Increased scaling)

             # Update previous distance to target for shaping
             self._prev_dist_to_target = distance_to_target_xy


        # Reward for placing a box in the target tray (existing logic)
        # Check if any box is within the target tray boundaries and not currently grasped
        for obj_id in self.object_ids:
             if obj_id not in self._boxes_in_target: # Only reward for boxes not already counted
                  obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
                  # Check if the object is within the target tray's approximate bounds
                  distance_to_target_center = np.linalg.norm(np.array(obj_pos) - np.array(self.TRAY_2_POS))

                  if distance_to_target_center < self.TARGET_POS_TOLERANCE and obj_pos[2] > (self.TRAY_2_POS[2] - self.TRAY_DIMENSIONS[2]/2): # Check if within tolerance and on/above tray surface
                       # Check if the box is stable (low velocity) and not currently grasped
                       lin_vel, ang_vel = p.getBaseVelocity(obj_id)
                       if np.linalg.norm(lin_vel) < 0.01 and not self._is_grasping: # If box is relatively still and not held
                            reward += self.REWARD_BOX_IN_TARGET # Reward for placing a box
                            self._boxes_in_target.append(obj_id) # Mark this box as placed
                            print(f"Box {obj_id} placed in target tray! Total placed: {len(self._boxes_in_target)}")


        # Additional reward for placing all boxes (existing logic)
        if len(self._boxes_in_target) == self.NUM_BOXES:
             reward += self.REWARD_ALL_BOXES_IN_TARGET


        # Optional: Add penalties for collisions or dropping boxes (existing logic)
        # Implementing robust collision detection and drop detection can be complex.
        # For now, we'll omit these to keep it simpler.
        # if self._check_collision():
        #      reward += REWARD_COLLISION_PENALTY
        # if self._check_dropped_box(): # Requires tracking if a grasped box was released unintentionally
        #      reward += REWARD_DROP_PENALTY


        return reward

    def _check_done(self):
        """
        Checks if the episode is finished.

        Returns:
            terminated: Boolean, True if the episode terminated (e.g., task completed or failed).
            truncated: Boolean, True if the episode was truncated (e.g., time limit reached).
        """
        terminated = False
        truncated = False

        # Check for task completion: all boxes in the target tray
        if len(self._boxes_in_target) == self.NUM_BOXES:
             print("All boxes placed. Task completed!")
             terminated = True

        # Check for time limit
        if self._env_step_counter >= self.MAX_ENVIRONMENT_STEPS:
             # print(f"Time limit reached ({self._env_step_counter} steps). Episode truncated.") # Commented out to reduce spam
             truncated = True

        # Optional: Add other termination conditions (e.g., robot tipped over, box dropped out of bounds)
        # if self._check_robot_tipped():
        #      terminated = True
        # if self._check_box_out_of_bounds():
        #      terminated = True


        return terminated, truncated

    def get_observation(self):
        """
        Gathers the current observation of the environment for the RL agent.

        Returns:
            obs: A dictionary containing the observation data:
                 'robot_joint_positions': List of robot joint positions.
                 'robot_joint_velocities': List of robot joint velocities.
                 'robot_ee_position': List [x, y, z] of end-effector position.
                 'robot_ee_orientation_euler': List [roll, pitch, yaw] of end-effector orientation.
                 'gripper_opening_length': Float, the current gripper opening length.
                 'object_states': NumPy array of flattened object poses (pos + euler for each object).
                 'rgb': RGB camera image (NumPy array) or None if camera is not active.
                 'depth': Depth camera image (NumPy array) or None if camera is not active.
                 'seg': Segmentation camera image (NumPy array) or None if camera is not active.
        """
        # Get robot state from the robot instance
        robot_obs = self.robot.get_joint_obs()
        robot_joint_positions = robot_obs['positions']
        robot_joint_velocities = robot_obs['velocities']
        robot_ee_position = robot_obs['ee_pos']

        # Get robot end-effector orientation (convert from quaternion to Euler)
        # Need to get the link state of the EE link to get its orientation
        ee_state = p.getLinkState(self.robot.id, self.robot.eef_id)
        robot_ee_orientation_quat = ee_state[1] # Orientation as quaternion
        robot_ee_orientation_euler = p.getEulerFromQuaternion(robot_ee_orientation_quat) # Convert to Euler (roll, pitch, yaw)

        # Get gripper opening length
        # Based on robot.py, the Robotiq 85 uses a mimic_parent_id for gripper control.
        try:
            # Get the state of the mimic parent joint for the gripper
            gripper_joint_state = p.getJointState(self.robot.id, self.robot.mimic_parent_id)
            # The position of this joint relates to the gripper opening.
            # Need to convert this joint position back to opening length.
            # This conversion is specific to the Robotiq 85 geometry and mimic setup.
            # From robot.py move_gripper: open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
            # We need the inverse: open_length = math.sin(0.715 - gripper_joint_state[0]) * 0.1143 + 0.010
            gripper_opening_length = math.sin(0.715 - gripper_joint_state[0]) * 0.1143 + 0.010
        except AttributeError:
            # Fallback if mimic_parent_id is not defined (e.g., for a different gripper)
            # In this case, we might need a different way to get gripper opening,
            # or just use the state of the last controllable joint if it's directly the opening.
            # For Robotiq 85, using the mimic parent is more accurate.
            # If using a different gripper, this part might need significant changes.
            # print("Warning: Could not get gripper state using mimic_parent_id. Gripper observation might be inaccurate.") # Commented out to reduce spam
            # As a fallback, maybe use the position of the last controllable joint,
            # assuming it's directly related to opening. This is a less reliable approach.
            if self.robot.controllable_joints:
                 gripper_opening_length = p.getJointState(self.robot.id, self.robot.controllable_joints[-1])[0]
            else:
                 gripper_opening_length = 0.0 # Default to 0 if no controllable joints


        # Get object states (positions and orientations) and flatten them
        object_states_flat = []
        for obj_id in self.object_ids:
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            object_states_flat.extend(pos) # Add position (x, y, z)
            # Convert quaternion to Euler angles for orientation
            object_states_flat.extend(p.getEulerFromQuaternion(orn)) # Add orientation (roll, pitch, yaw)

        # Get camera data if camera is active
        rgb = None
        depth = None
        seg = None
        if isinstance(self.camera, Camera):
             rgb, depth, seg = self.camera.shot()


        # Return observation as a dictionary
        observation = {
            'robot_joint_positions': np.array(robot_joint_positions),
            'robot_joint_velocities': np.array(robot_joint_velocities),
            'robot_ee_position': np.array(robot_ee_position),
            'robot_ee_orientation_euler': np.array(robot_ee_orientation_euler),
            'gripper_opening_length': np.array([gripper_opening_length]),
            'object_states': np.array(object_states_flat),
            'rgb': rgb,
            'depth': depth,
            'seg': seg
        }

        return observation

    def reset(self):
        """
        Resets the environment to an initial state by spawning objects.
        Resets the robot, removes existing objects, and spawns new objects.
        Resets reward and done related state variables.
        """
        # Reset the robot to its home/rest pose and open the gripper
        self.robot.reset()

        # Remove any existing objects from the previous episode
        for obj_id in self.object_ids:
            p.removeBody(obj_id)
        self.object_ids.clear() # Clear the list of object IDs
        # Reset state variables related to reward and done
        self._boxes_in_target.clear()
        self._is_grasping = False
        self._prev_is_grasping = False # Reset previous grasping state
        # Reset variables used for reward shaping
        self._prev_dist_to_object = None
        self._prev_dist_to_target = None


        # Define the area within tray 1 where objects can be spawned
        # Ensure objects are spawned slightly above the tray surface to avoid initial collisions
        # Using the dimensions of the tray's base collision box for calculating spawn area
        # Spawn area is relative to the TRAY_1_POS (which is on top of Table 1)
        min_x = self.TRAY_1_POS[0] - self.TRAY_DIMENSIONS[0] / 2 + self.BOX_SIDE_LENGTH / 2
        max_x = self.TRAY_1_POS[0] + self.TRAY_DIMENSIONS[0] / 2 - self.BOX_SIDE_LENGTH / 2
        min_y = self.TRAY_1_POS[1] - self.TRAY_DIMENSIONS[1] / 2 + self.BOX_SIDE_LENGTH / 2
        max_y = self.TRAY_1_POS[1] + self.TRAY_DIMENSIONS[1] / 2 - self.BOX_SIDE_LENGTH / 2
        # Adjust spawn_z higher to reduce initial collisions
        spawn_z = self.TRAY_1_POS[2] + 0.2 # Spawn higher above the tray surface

        # Create visual and collision shapes for the box programmatically
        box_half_extents = [self.BOX_SIDE_LENGTH / 2.0] * 3 # Half extents for a cube
        box_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=box_half_extents,
                                                  rgbaColor=[0.4, 0.4, 1.0, 1.0]) # Blue color
        box_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                        halfExtents=box_half_extents)

        # Spawn new objects (cubes) randomly within the defined area in tray 1
        for i in range(self.NUM_BOXES): # Use the class constant for the number of boxes
            # Generate random x, y position within the tray bounds
            spawn_pos = [random.uniform(min_x, max_x),
                         random.uniform(min_y, max_y),
                         spawn_z]
            # Generate random orientation (optional, starting upright is usually easier)
            # spawn_orn = p.getQuaternionFromEuler([random.uniform(0, np.pi), random.uniform(0, np.pi), random.uniform(0, np.pi)])
            # Start with an upright orientation
            spawn_orn = p.getQuaternionFromEuler([0, 0, 0])

            # Create the multibody for the box
            obj_id = p.createMultiBody(baseMass=self.BOX_MASS, # Set the mass
                                       baseCollisionShapeIndex=box_collision_shape_id,
                                       baseVisualShapeIndex=box_visual_shape_id,
                                       basePosition=spawn_pos,
                                       baseOrientation=spawn_orn)

            self.object_ids.append(obj_id) # Add the spawned object's ID to the list

            # Optional: change other dynamics properties (friction, restitution) if needed
            # p.changeDynamics(obj_id, -1, lateralFriction=0.8, restitution=0.1)


        # Let objects settle onto the tray surface by running simulation steps
        # Increased settling steps to allow more time for objects to stabilize
        for _ in range(200): # Run 200 simulation steps for settling
            self.step_simulation()

        # Reset environment step counter
        self._env_step_counter = 0

        print(f"Environment reset complete. Spawned {len(self.object_ids)} objects.")
        # Return the initial observation of the reset environment
        return self.get_observation()

    def close(self):
        """
        Disconnects from the PyBullet physics server.
        """
        p.disconnect(self.physicsClient)

