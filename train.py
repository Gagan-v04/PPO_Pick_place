# train_ppo.py
# This script sets up and trains a PPO agent using Stable Baselines3
# on the PickPlaceEnv with state-based observations.
# Runs in headless mode for faster execution and trains for a specified number of timesteps.
# Enables TensorBoard logging to monitor training progress.
# Configured for multi-core usage using SubprocVecEnv.
# Includes functionality to load a previously trained agent and continue training.
# Added a custom callback to track and log the maximum episode reward.

import gymnasium as gym # Using Gymnasium for environment compatibility
import numpy as np
import pybullet as p
import pybullet_data
import time
import math # Import math for the gripper conversion in get_observation
import os # Import os for creating directories

from stable_baselines3 import PPO # Import the PPO algorithm
# Import utility function to create vectorized environments and the multiprocessing vec env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv # Using SubprocVecEnv for multiprocessing
# Import BaseCallback for creating custom callbacks
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Removed unused vectorized environment imports for clarity
# from stable_baselines3.common.vec_env import DummyVecEnv # No longer using DummyVecEnv
# from stable_baselines3.common.env_checker import check_env # Not typically used directly with vec envs

# Import your custom environment and robot classes
from env import PickPlaceEnv
from robot import UR5Robotiq85
from utilities import Camera # Keep import even if not used for observation yet

# --- Custom Callback to Log Max Episode Reward ---
class MaxEpisodeRewardCallback(BaseCallback):
    """
    A custom callback to track and log the maximum episode reward encountered during training.
    """
    def __init__(self, verbose=0):
        super(MaxEpisodeRewardCallback, self).__init__(verbose)
        self.max_episode_reward = -float('inf') # Initialize with a very low value
        self._last_log_step = 0 # To control logging frequency

    def _on_step(self) -> bool:
        """
        This method is called after each environment step.
        """
        # Check if any episode has finished in the parallel environments
        # infos is a list of dictionaries, one for each environment
        for info in self.locals['infos']:
            # Check if the 'episode' key exists in the info dictionary
            # This key is added by the VecEnvWrapper when an episode finishes
            if 'episode' in info:
                episode_reward = info['episode']['r'] # Get the reward for the finished episode
                # Update the maximum episode reward if the current one is higher
                if episode_reward > self.max_episode_reward:
                    self.max_episode_reward = episode_reward
                    if self.verbose > 0:
                        print(f"New max episode reward: {self.max_episode_reward:.2f}")

        # Log the maximum episode reward to TensorBoard periodically
        # We log based on the total number of steps taken by the agent (self.num_timesteps)
        log_interval = 1000 # Log every 1000 steps (adjust as needed)
        if (self.num_timesteps - self._last_log_step) >= log_interval:
             self.logger.record('rollout/max_ep_rew', self.max_episode_reward)
             self._last_log_step = self.num_timesteps


        # Return True to continue training, False to stop training
        return True

    def _on_training_start(self) -> None:
        """
        This method is called before the first call to _on_step().
        """
        # Optional: Log initial values or setup
        pass

    def _on_training_end(self) -> None:
        """
        This method is called before exiting the `learn()` method.
        """
        # Optional: Log final values or cleanup
        pass

# --- Environment Definition (Wrapper for Gymnasium) ---
# We need to wrap our PickPlaceEnv to make it compatible with Gymnasium and Stable Baselines3.
# This involves defining the observation_space and action_space attributes.

class PickPlaceEnvGym(gym.Env):
    """
    A Gymnasium wrapper for the PickPlaceEnv to be compatible with Stable Baselines3.
    Uses state-based observations and joint control actions.
    """
    # Define required attributes for Gymnasium API
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240} # Example metadata
    reward_range = (-float('inf'), float('inf')) # Define potential reward range
    spec = None # Can be set later if needed

    def __init__(self, robot, vis=False):
        super(PickPlaceEnvGym, self).__init__()

        # Instantiate the original PickPlaceEnv
        # Pass vis=False when using vectorized environments for faster training
        self._env = PickPlaceEnv(robot=robot, vis=vis)

        # --- Define Observation Space ---
        # The observation space will be a Box space representing the flattened state vector
        # from PickPlaceEnv.get_observation().
        # We need to determine the size of this flattened vector based on the actual data shapes.
        # Based on debug output (and env.py):
        # - robot_joint_positions: 12
        # - robot_joint_velocities: 12
        # - robot_ee_position: 3 (x, y, z)
        # - robot_ee_orientation_euler: 3 (roll, pitch, yaw)
        # - gripper_opening_length: 1
        # - object_states: NUM_BOXES * (3 pos + 3 euler) = NUM_BOXES * 6
        # Total size = 12 + 12 + 3 + 3 + 1 + (self._env.NUM_BOXES * 6)
        observation_size = (12 + 12 + 3 + 3 + 1) + (self._env.NUM_BOXES * 6)

        # Define the bounds for the observation space.
        # Ensure bounds match the flattened observation size.
        # Order of flattened observation:
        # [robot_joint_positions (12), robot_joint_velocities (12),
        #  robot_ee_position (3), robot_ee_orientation_euler (3),
        #  gripper_opening_length (1),
        #  object1_pos (3), object1_euler (3),
        #  object2_pos (3), object2_euler (3),
        #  object3_pos (3), object3_euler (3),
        #  object4_pos (3), object4_euler (3)]

        # Bounds for robot state
        # Adjusted bounds for robot joint positions as requested
        robot_joint_pos_low = [-3.15] * 12
        robot_joint_pos_high = [3.15] * 12
        robot_joint_vel_low = [-10.] * 12 # Approximate velocity limits
        robot_joint_vel_high = [10.] * 12
        # Increased the bounds for robot_ee_position to prevent AssertionErrors
        # based on observed end-effector positions during execution.
        # You may need to adjust these bounds further depending on the robot's actual workspace.
        robot_ee_pos_low = [-2.0] * 3 # Example: Increased lower bound
        robot_ee_pos_high = [2.0] * 3 # Example: Increased upper bound
        robot_ee_euler_low = [-np.pi] * 3
        robot_ee_euler_high = [np.pi] * 3
        gripper_low = [0.0]
        gripper_high = [self._env.robot.gripper_range[1]]

        # Bounds for object states (repeated for each object)
        # Assuming objects stay within a reasonable workspace area around the tables.
        # Adjust if objects can be moved further away.
        object_pos_low = [-1.5] * 3 # Approximate workspace limits for objects
        object_pos_high = [1.5] * 3 # Approximate workspace limits for objects
        object_euler_low = [-np.pi] * 3 # Orientation bounds for objects
        object_euler_high = [np.pi] * 3

        # Construct the full low and high arrays in the correct order
        low = np.concatenate([
            robot_joint_pos_low,
            robot_joint_vel_low,
            robot_ee_pos_low,
            robot_ee_euler_low,
            gripper_low,
            np.tile(np.concatenate([object_pos_low, object_euler_low]), self._env.NUM_BOXES) # Repeat pos and euler bounds for each object
        ])

        high = np.concatenate([
            robot_joint_pos_high,
            robot_joint_vel_high,
            robot_ee_pos_high,
            robot_ee_euler_high,
            gripper_high,
            np.tile(np.concatenate([object_pos_high, object_euler_high]), self._env.NUM_BOXES) # Repeat pos and euler bounds for each object
        ])


        # Ensure the calculated size matches the bounds size
        assert observation_size == low.shape[0] == high.shape[0], f"Observation space size mismatch. Calculated: {observation_size}, Low/High shape: {low.shape[0]}"

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Define Action Space ---
        # The action space will be a Box space representing the target joint angles
        # and the target gripper opening length.
        # Action size = number of controllable arm joints + 1 (for gripper)
        action_size = self._env.robot.arm_num_dofs + 1 # 6 arm joints + 1 gripper (for UR5)

        # Define the bounds for the action space.
        # Joint angle bounds: Use robot's joint limits
        # Gripper opening bound: Use robot's gripper range
        action_low = np.array(self._env.robot.arm_lower_limits + [self._env.robot.gripper_range[0]])
        action_high = np.array(self._env.robot.arm_upper_limits + [self._env.robot.gripper_range[1]])

        # Ensure the calculated size matches the bounds size
        assert action_size == action_low.shape[0] == action_high.shape[0], "Action space size mismatch"

        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Store the robot instance (primarily for accessing robot properties like gripper_range)
        self.robot = self._env.robot


    def step(self, action):
        """
        Steps the environment using the given action.

        Args:
            action: A NumPy array representing the target joint angles and gripper opening.

        Returns:
            observation: The new observation after the step.
            reward: The reward for the step.
            terminated: True if the episode terminated (e.g., task completed or failed).
            truncated: True if the episode was truncated (e.g., time limit reached).
            info: Additional information.
        """
        # The action from the RL agent is a NumPy array.
        # The PickPlaceEnv.step expects a list or tuple.
        action_list = action.tolist()

        # Step the underlying PickPlaceEnv using joint control
        # The _env.step method returns obs, reward, terminated, truncated, info (5 values)
        # Correctly unpack the 5 values returned by the underlying environment
        observation_dict, reward, terminated, truncated, info = self._env.step(action_list, control_method='joint')

        # Flatten the observation dictionary into a NumPy array for the RL agent
        # Ensure the order matches the observation_space definition
        # Exclude camera data as it's not used in the flattened state observation
        observation = np.concatenate([
            observation_dict['robot_joint_positions'],
            observation_dict['robot_joint_velocities'],
            observation_dict['robot_ee_position'],
            observation_dict['robot_ee_orientation_euler'],
            observation_dict['gripper_opening_length'],
            observation_dict['object_states']
        ]).astype(np.float32) # Ensure dtype is float32 as required by Gym spaces

        # In Gymnasium, the step method returns (observation, reward, terminated, truncated, info)
        # We already have terminated and truncated from the underlying env.
        # info is also returned by the underlying env.
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment.

        Args:
            seed: An optional seed for the random number generator.
            options: Additional options for resetting.

        Returns:
            observation: The initial observation after reset.
            info: Additional information.
        """
        # Call the parent class reset. This is important for seeding in vectorized environments.
        super().reset(seed=seed)

        # Reset the underlying PickPlaceEnv
        observation_dict = self._env.reset()

        # Flatten the initial observation dictionary
        # Exclude camera data as it's not used in the flattened state observation
        observation = np.concatenate([
            observation_dict['robot_joint_positions'],
            observation_dict['robot_joint_velocities'],
            observation_dict['robot_ee_position'],
            observation_dict['robot_ee_orientation_euler'],
            observation_dict['gripper_opening_length'],
            observation_dict['object_states']
        ]).astype(np.float32) # Ensure dtype is float32

        # Return observation and info (info can be empty initially)
        info = {} # Populate with reset-specific info if needed

        # Stable-Baselines3 expects the seed to be returned in the info dict during reset
        # when using vectorized environments with seeding.
        # See: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vec-envs-specifics
        if seed is not None:
             info["seed"] = seed

        return observation, info

    def render(self):
        """
        Renders the environment (handled by PyBullet GUI if vis=True).
        """
        # Rendering is handled by the PyBullet GUI when self._env.vis is True
        # Note: Rendering is typically disabled when using vectorized environments for training speed.
        pass

    def close(self):
        """
        Closes the environment.
        """
        self._env.close()

# Create a function that returns an environment instance
# This is required for make_vec_env
def make_env():
    """Helper function to create a single environment instance."""
    robot_base_pos = [0, 0, 0.63]
    robot_base_ori = p.getQuaternionFromEuler([0, 0, 0]) # Use quaternion for robot base orientation
    # Instantiate the robot for this environment instance
    robot_instance = UR5Robotiq85(robot_base_pos, p.getEulerFromQuaternion(robot_base_ori))
    # Return the Gymnasium-wrapped environment
    # Set vis=False to run in headless mode for multiprocessing
    return PickPlaceEnvGym(robot=robot_instance, vis=False)


# --- Main Training Script ---

if __name__ == '__main__':
    # --- Environment Setup ---
    # Create a vectorized environment to run multiple instances in parallel
    # The number of environments (n_envs) determines how many CPU cores will be utilized
    # Set n_envs to the number of cores available (e.g., 4 for i5-8250U)
    num_cpu = 4 # Number of environments to run in parallel

    print(f"Creating {num_cpu} parallel environments using SubprocVecEnv...")
    # Use SubprocVecEnv for multiprocessing to distribute workload across CPU cores
    # make_vec_env handles creating multiple instances using the make_env function
    # and automatically wraps them in a SubprocVecEnv.
    env = make_vec_env(make_env, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    print("Vectorized environment created.")


    # Optional: Check a single instance of the environment for compliance
    # Note: check_env is typically used for a single environment, not a vectorized one.
    # You can get an unwrapped env from the vec env if needed for checking.
    # try:
    #     print("Checking a single environment instance compliance...")
    #     # Access an unwrapped environment from the vectorized environment for checking
    #     # You might need to adjust the index [0] if using a different VecEnv type
    #     check_env(env.get_attr("_env")[0], warn=True)
    #     print("Single environment instance check passed.")
    # except Exception as e:
    #      print(f"Error during single environment compliance check: {e}")


    # --- Agent Setup ---
    agent_save_path = "./ppo_pick_place_agent"
    log_dir = "./ppo_pick_place_tensorboard/"
    os.makedirs(log_dir, exist_ok=True) # Create the directory if it doesn't exist

    # Check if a trained agent already exists to continue training
    if os.path.exists(agent_save_path + ".zip"):
        print(f"Loading existing trained agent from {agent_save_path}.zip")
        # Load the model, making sure to pass the environment and other necessary parameters
        # Pass the vectorized environment 'env' to the loaded model
        model = PPO.load(agent_save_path, env, tensorboard_log=log_dir, verbose=1)
        print("Agent loaded successfully. Continuing training.")
    else:
        print("No existing agent found. Creating a new agent.")
        # Define the policy network architecture.
        # MlpPolicy is suitable for state-based observations.
        # You can customize the network size (e.g., net_arch=[64, 64]).
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64])) # Example: two layers of 64 units for actor and critic

        # Instantiate a new PPO agent
        model = PPO("MlpPolicy",
                    env, # Pass the vectorized environment
                    learning_rate=0.0003,
                    n_steps=2048, # Number of steps to run in *each* environment per update. Total steps per update = n_steps * n_envs
                    batch_size=64, # Minibatch size for training
                    n_epochs=10, # Number of epochs to update the policy
                    gamma=0.99, # Discount factor
                    gae_lambda=0.95, # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
                    clip_range=0.2, # Clipping parameter for PPO
                    ent_coef=0.01, # Entropy coefficient for exploration
                    verbose=1, # Set to 1 or 2 for more training information
                    tensorboard_log=log_dir # Enabled TensorBoard logging
                   )
        print("New agent created.")

    # --- Setup Callback(s) ---
    # Instantiate the custom callback to track max episode reward
    max_reward_callback = MaxEpisodeRewardCallback(verbose=1) # Set verbose=1 to print updates

    # If you have multiple callbacks, you can combine them into a CallbackList
    # callbacks = CallbackList([max_reward_callback, other_callback_instance])
    # For now, we just have one callback
    callbacks = max_reward_callback


    # --- Training ---
    print("Starting/Continuing training...")
    # Train the agent for a specified number of timesteps
    # The total timesteps is accumulated across all parallel environments.
    # Set the number of *additional* timesteps you want to train for in this run
    additional_timesteps = 500000 # Changed to 500,000

    # The learn method will add these additional_timesteps to the model's internal timestep count
    # Use reset_num_timesteps=False to continue the total timestep count from the loaded model
    # Pass the callback(s) to the learn method
    model.learn(total_timesteps=additional_timesteps, reset_num_timesteps=False, callback=callbacks)
    print(f"Training finished after adding {additional_timesteps} timesteps.")
    # The total timesteps reached will be the sum of previous training + additional_timesteps


    # --- Save the trained agent ---
    # This will overwrite the previous save file with the updated model
    model.save(agent_save_path)
    print(f"Agent saved to {agent_save_path}.zip")


    # Close the environment
    # This will close all parallel environments created by make_vec_env
    env.close()
    print("Environments closed.")

