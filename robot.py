# Modified robot.py
# Defines base class and specific implementations for robots in the PyBullet environment.
# Includes methods for loading the robot, resetting, moving the arm and gripper,
# and getting joint observations.

import pybullet as p
import math
import numpy as np # Added numpy import for potential use
from collections import namedtuple


class RobotBase(object):
    """
    The base class for robots in the PyBullet environment.
    Provides common functionality for loading, controlling, and observing robots.
    """

    def __init__(self, pos, ori):
        """
        Initializes the RobotBase with base position and orientation.

        Args:
            pos: List or tuple [x, y, z], the base position of the robot in world coordinates.
            ori: List or tuple [r, p, y], the base orientation of the robot in Euler angles (roll, pitch, yaw).

        Attributes:
            id: Int, the unique ID assigned to the robot model in PyBullet.
            eef_id: Int, the link ID of the End-Effector.
            arm_num_dofs: Int, the number of degrees of freedom (controllable joints) in the robot arm.
                Inverse Kinematics (IK) calculations will typically consider these joints.
            joints: List of namedtuples, containing information about each joint in the robot model.
            controllable_joints: List of Ints, IDs for all controllable (non-fixed) joints.
            arm_controllable_joints: List of Ints, IDs for the first `arm_num_dofs` controllable joints (the arm joints).

            ---
            For null-space IK (used in calculateInverseKinematics)
            ---
            arm_lower_limits: List of floats, the lower joint limits for arm_controllable_joints.
            arm_upper_limits: List of floats, the upper joint limits for arm_controllable_joints.
            arm_joint_ranges: List of floats, the range (upper - lower) for arm_controllable_joints.
            arm_rest_poses: List of floats, the preferred rest position for arm_controllable_joints (used in IK).

            gripper_range: List [Min, Max], the minimum and maximum values for gripper control (e.g., open_length).
        """
        self.base_pos = pos # Base position [x, y, z]
        # Convert Euler orientation to quaternion for PyBullet
        self.base_ori = p.getQuaternionFromEuler(ori) # Base orientation [qx, qy, qz, qw]

        # Initialize attributes that will be populated after loading the robot
        self.id = None
        self.eef_id = None
        self.arm_num_dofs = None
        self.joints = []
        self.controllable_joints = []
        self.arm_controllable_joints = []
        self.arm_lower_limits = []
        self.arm_upper_limits = []
        self.arm_joint_ranges = []
        self.arm_rest_poses = []
        self.gripper_range = [0, 0] # Default gripper range

        # This method will be hooked by the environment to step the simulation
        self.step_simulation = None # Placeholder

    def load(self):
        """
        Loads the robot URDF into the PyBullet simulation.
        Initializes joint information and performs post-load setup.
        """
        self.__init_robot__() # Robot-specific initialization (loads URDF)
        self.__parse_joint_info__() # Parse joint details from the loaded model
        self.__post_load__() # Robot-specific post-load setup (e.g., mimic joints)
        # print("Robot Joints Info:") # Optional: print joint info after loading
        # for joint in self.joints:
        #     print(joint)

    def step_simulation(self):
        """
        Placeholder for the simulation step function.
        This method should be assigned by the environment instance.
        """
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')

    def __parse_joint_info__(self):
        """
        Parses joint information from the loaded robot model.
        Identifies controllable joints and extracts limits and ranges.
        """
        numJoints = p.getNumJoints(self.id) # Get the total number of joints
        # Define a namedtuple to store joint information
        jointInfo = namedtuple('jointInfo',
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = [] # List to store joint information
        self.controllable_joints = [] # List to store IDs of controllable joints
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i) # Get information for joint i
            jointID = info[0]
            jointName = info[1].decode("utf-8") # Decode joint name from bytes
            jointType = info[2]  # Joint type (JOINT_REVOLUTE, JOINT_PRISMATIC, etc.)
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            # A joint is controllable if it's not of type JOINT_FIXED
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID) # Add to controllable joints list
                # Set initial motor control for controllable joints to velocity control with zero force
                # This prevents uncontrolled movement before position commands are sent.
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            # Create a jointInfo namedtuple and add it to the joints list
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        # Ensure the number of controllable joints is at least the defined arm_num_dofs
        assert len(self.controllable_joints) >= self.arm_num_dofs, "Number of controllable joints is less than arm_num_dofs"
        # The arm controllable joints are the first arm_num_dofs controllable joints
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        # Extract lower limits, upper limits, and ranges for the arm controllable joints
        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        # Calculate joint ranges (upper limit - lower limit)
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        # Note: arm_rest_poses should be set in the robot-specific __init_robot__ or __post_load__

    def __init_robot__(self):
        """
        Robot-specific initialization, including loading the URDF.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __post_load__(self):
        """
        Robot-specific post-load setup, such as setting up mimic joints.
        Can be overridden by subclasses.
        """
        pass # Default implementation does nothing

    def reset(self):
        """
        Resets the robot to a default state (rest pose for arm, open gripper).
        """
        self.reset_arm() # Reset arm joints
        self.reset_gripper() # Reset gripper

    def reset_arm(self):
        """
        Resets the robot arm joints to their defined rest poses.
        """
        # Iterate through arm controllable joints and reset their state to the rest pose
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)

        # Run a few simulation steps to allow the robot to settle after resetting
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        """
        Resets the gripper to its open position.
        """
        self.open_gripper()

    def open_gripper(self):
        """
        Commands the gripper to its fully open position.
        """
        self.move_gripper(self.gripper_range[1]) # Move to the maximum gripper range

    def close_gripper(self):
        """
        Commands the gripper to its fully closed position.
        """
        self.move_gripper(self.gripper_range[0]) # Move to the minimum gripper range

    def move_ee(self, action, control_method):
        """
        Moves the robot arm based on the specified action and control method.

        Args:
            action: A tuple or list representing the desired arm state.
                    - 'end': (x, y, z, roll, pitch, yaw) for target end-effector pose.
                    - 'joint': List of target joint positions for arm_controllable_joints.
            control_method: String, 'end' for end-effector control, 'joint' for joint position control.
        """
        assert control_method in ('joint', 'end'), "Invalid control_method. Use 'joint' or 'end'."
        if control_method == 'end':
            # End-effector position control using Inverse Kinematics (IK)
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z) # Target position
            orn = p.getQuaternionFromEuler((roll, pitch, yaw)) # Target orientation (as quaternion)
            # Calculate joint positions using IK
            # Uses arm_lower_limits, arm_upper_limits, arm_joint_ranges, and arm_rest_poses for null-space control/IK solving
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses,
                                                       maxNumIterations=20) # Max iterations for IK solver
        elif control_method == 'joint':
            # Joint position control
            assert len(action) == self.arm_num_dofs, f"Joint action must have {self.arm_num_dofs} values."
            joint_poses = action # Target joint positions directly provided

        # Apply the calculated or provided joint positions using position control
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, # Use max force defined in URDF
                                    maxVelocity=self.joints[joint_id].maxVelocity) # Use max velocity defined in URDF


    def move_gripper(self, open_length):
        """
        Commands the gripper to a specific opening length.
        Must be implemented by subclasses as gripper control is robot-specific.
        """
        raise NotImplementedError

    def get_joint_obs(self):
        """
        Gets the current state observation of the robot's controllable joints and end-effector.

        Returns:
            Dictionary containing:
            - 'positions': List of current positions for all controllable joints.
            - 'velocities': List of current velocities for all controllable joints.
            - 'ee_pos': List [x, y, z], the current position of the end-effector link.
        """
        positions = []
        velocities = []
        # Iterate through all controllable joints to get their state
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        # Get the current position of the end-effector link
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)


# --- Specific Robot Implementations ---

class Panda(RobotBase):
    """
    Implementation for the Franka Emika Panda robot.
    """
    def __init_robot__(self):
        # Define end-effector link ID and number of arm DoFs
        self.eef_id = 11
        self.arm_num_dofs = 7
        # Define rest poses for arm joints
        self.arm_rest_poses = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]
        # Load the Panda URDF model
        self.id = p.loadURDF('./urdf/panda.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, # Robot base is fixed in the world
                             flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES) # Optimize graphics loading
        # Define the gripper range (min, max opening length)
        self.gripper_range = [0, 0.04] # Example range

        # Create a constraint to keep the Panda fingers synchronized
        # This is specific to the Panda URDF and how its fingers are defined.
        c = p.createConstraint(self.id,
                               9, # Parent link ID (finger joint 1)
                               self.id,
                               10, # Child link ID (finger joint 2)
                               jointType=p.JOINT_GEAR, # Gear constraint
                               jointAxis=[1, 0, 0], # Axis of rotation for the gear relationship
                               parentFramePosition=[0, 0, 0], # Position in parent frame
                               childFramePosition=[0, 0, 0]) # Position in child frame
        # Configure the gear constraint: gearRatio=-1 means child moves opposite to parent
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    def move_gripper(self, open_length):
        """
        Commands the Panda gripper to a specific opening length.
        """
        # Clamp the desired opening length to the valid gripper range
        open_length = np.clip(open_length, *self.gripper_range)
        # Control the finger joints directly based on the desired opening length
        for i in [9, 10]: # Joint IDs for the two fingers
            p.setJointMotorControl2(self.id, i, p.POSITION_CONTROL, open_length, force=20) # Apply position control


class UR5Robotiq85(RobotBase):
    """
    Implementation for the Universal Robots UR5 with a Robotiq 85 gripper.
    """
    def __init_robot__(self):
        # Define end-effector link ID and number of arm DoFs
        self.eef_id = 7 # Link ID for the wrist_3_link (usually the end-effector attachment point)
        self.arm_num_dofs = 6 # UR5 has 6 revolute joints in the arm
        # Define rest poses for arm joints (in radians)
        # These values are typical for a home/start configuration.
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        # Load the UR5 with Robotiq 85 gripper URDF model
        # Assuming the URDF is located at './urdf/ur5_robotiq_85.urdf'
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, # Robot base is fixed
                             flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        # Define the gripper range (min, max opening length in meters)
        self.gripper_range = [0, 0.085] # Robotiq 85 nominal range

    def __post_load__(self):
        """
        Post-load setup for the UR5 Robotiq 85, specifically setting up mimic joints for the gripper.
        """
        # Define the parent joint name for mimic constraints
        mimic_parent_name = 'finger_joint' # This joint directly controls the gripper opening
        # Define child joint names and their multiplier relative to the parent
        # Robotiq 85 fingers move in a specific coordinated way
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1, # Inner fingers move opposite to outer
                                'right_inner_finger_joint': -1}
        # Set up the mimic constraints
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        """
        Sets up mimic constraints for gripper joints.
        One joint (parent) is directly controlled, and others (children) follow its movement.
        """
        # Find the ID of the parent joint by its name
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        # Create a dictionary mapping child joint IDs to their multiplier
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        # Create gear constraints for each child joint
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id, # Parent body and joint
                                   self.id, joint_id, # Child body and joint
                                   jointType=p.JOINT_GEAR, # Gear constraint type
                                   jointAxis=[0, 1, 0], # Axis for the gear relationship (depends on URDF joint axis)
                                   parentFramePosition=[0, 0, 0], # Position in parent frame
                                   childFramePosition=[0, 0, 0]) # Position in child frame
            # Configure the gear constraint: gearRatio=-multiplier ensures child follows parent with multiplier
            # erp (Error Reduction Parameter) is important for stability
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def move_gripper(self, open_length):
        """
        Commands the Robotiq 85 gripper to a specific opening length.
        Converts the desired opening length to the required angle for the parent mimic joint.
        """
        # Clamp the desired opening length to the valid gripper range
        # open_length = np.clip(open_length, *self.gripper_range) # Already handled by debug param range

        # Calculate the required angle for the 'finger_joint' based on the desired opening length
        # This formula is specific to the Robotiq 85 gripper geometry.
        # Source: https://github.com/ros-industrial/robotiq/blob/kinetic-devel/robotiq_85_description/urdf/robotiq_85_gripper.urdf.xacro
        # The angle is relative to the joint's zero position.
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143) # angle calculation

        # Control the parent mimic joint ('finger_joint') using position control
        # The child joints will follow due to the gear constraints.
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, # Use max force
                                maxVelocity=self.joints[self.mimic_parent_id].maxVelocity) # Use max velocity


class UR5Robotiq140(UR5Robotiq85):
    """
    Implementation for the Universal Robots UR5 with a Robotiq 140 gripper.
    Inherits from UR5Robotiq85 as the arm is the same, but overrides gripper specifics.
    """
    def __init_robot__(self):
        # Define end-effector link ID and number of arm DoFs (same as UR5)
        self.eef_id = 7
        self.arm_num_dofs = 6
        # Define rest poses for arm joints (same as UR5)
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        # Load the UR5 with Robotiq 140 gripper URDF model
        # Assuming the URDF is located at './urdf/ur5_robotiq_140.urdf'
        self.id = p.loadURDF('./urdf/ur5_robotiq_140.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True,
                             flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        # Define the gripper range for Robotiq 140 (min, max opening length)
        self.gripper_range = [0, 0.140] # Robotiq 140 nominal range is 140mm = 0.140m
        # TODO: The angle calculation formula in move_gripper might need adjustment for Robotiq 140,
        # as the geometry might differ from the 85. Using the same formula as 85 for now.

    def __post_load__(self):
        """
        Post-load setup for the UR5 Robotiq 140, setting up mimic joints.
        Note: Multipliers might be different compared to Robotiq 85 depending on the URDF.
        """
        mimic_parent_name = 'finger_joint'
        # Multipliers for Robotiq 140 fingers might be inverted compared to 85
        mimic_children_names = {'right_outer_knuckle_joint': -1, # Example: Might move opposite
                                'left_inner_knuckle_joint': -1,
                                'right_inner_knuckle_joint': -1,
                                'left_inner_finger_joint': 1,
                                'right_inner_finger_joint': 1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    # Inherits move_gripper from UR5Robotiq85.
    # As noted above, the angle calculation formula might need adjustment for the Robotiq 140.
    # If the geometry is different, a new formula or approach would be needed here.
    # For now, it uses the same formula as the 85, which might not be accurate for the 140.

