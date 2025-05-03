Robotic Pick and Place with Reinforcement Learning

This repository contains code for training a robotic arm to perform a pick and place task in the PyBullet physics simulator using Reinforcement Learning (RL). The project utilizes the Stable Baselines3 library with the Proximal Policy Optimization (PPO) algorithm.

---

Project Structure

- train.py: Script for training the PPO agent. Also includes a mode for rendering the trained agent's behavior.
- env.py: Defines the custom PyBullet environment for the pick and place task, including object spawning, reward function, and termination conditions.
- robot.py: (Assumed to exist) Defines the robotic arm and gripper interface for interacting with PyBullet.
- utilities.py: (Assumed to exist) Contains utility classes or functions, such as the Camera class.
- ppo_pick_place_agent.zip: (Generated during training) The saved trained PPO model.
- ppo_pick_place_tensorboard/: (Generated during training) Directory containing TensorBoard logs for monitoring training progress.

---

Setup

Clone the repository:
git clone <repository_url>
cd <repository_name>

Install dependencies:
You will need Python 3.7 or higher. Install the required libraries using pip:

pip install gymnasium stable-baselines3 pybullet numpy

Note: You may need to install additional dependencies depending on your system and the specific versions of the libraries.

---

Usage

Training the Agent

To train the PPO agent, run the train.py script in train mode. By default, it will train for 500,000 additional timesteps:

python train.py --mode train

To specify a different number of training timesteps:

python train.py --mode train --timesteps 1000000

The script will automatically load a previously saved agent (ppo_pick_place_agent.zip) if it exists and continue training. TensorBoard logs will be saved in the ppo_pick_place_tensorboard/ directory.

---

Monitoring Training with TensorBoard

While training is in progress, you can monitor the learning process using TensorBoard:

tensorboard --logdir ppo_pick_place_tensorboard/

Then, open your browser and go to the provided address (usually http://localhost:6006/) to view plots such as episode reward, loss functions, and more.

---

Visualizing Agent Behavior

To visualize the trained agent in the PyBullet GUI, run:

python train.py --mode render

This will load the latest saved agent and run it in a single environment instance with the GUI enabled. Press Ctrl+C to stop the rendering.

---

Environment and Reward Function (env.py)

The PickPlaceEnv class defines a custom Gymnasium environment using PyBullet.

The reward function inside _calculate_reward() incentivizes:

- Approaching the target object
- Successfully grasping an object
- Lifting a grasped object
- Moving a grasped object towards the target tray
- Successfully placing the object in the tray
- Completing placement of all objects

It also includes penalties for:

- Each time step taken
- The end-effector going below the table surface

Refining the reward function is critical for optimal agent performance. Tune the reward scales based on visual observations and TensorBoard metrics.

---

Potential Improvements

- Reward Function Refinement: Add penalties for collisions or dropping objects, or better reward shaping.
- Observation Space: Use camera images (RGB, depth, segmentation) for more realistic inputs.
- Action Space: Explore control through end-effector velocity or impedance control.
- Hyperparameter Tuning: Adjust PPO settings such as learning rate, n_steps, batch_size, gamma, etc.
- Different RL Algorithms: Try SAC, TD3, or other algorithms from Stable Baselines3.
- Curriculum Learning: Gradually increase task difficulty.
- Domain Randomization: Vary environment parameters (positions, colors, friction) to improve generalization.

---

Feel free to fork and modify this project to suit your own robotic learning experiments.
