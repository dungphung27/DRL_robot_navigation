# Autonomous Robot Navigation Using Deep Q-Network (DQN)

## 1. Project Overview

This project investigates autonomous robot navigation using Deep Reinforcement Learning.

The goal is to train a mobile robot to navigate toward a target while avoiding obstacles in a two-dimensional environment.

Instead of following a predefined path, the robot learns how to choose actions through interactions with the environment.

The reinforcement learning algorithm used in this project is **Deep Q-Network (DQN)**, implemented using Python and PyTorch.

This project was developed as an individual Deep Learning course project.

## 2. Project Objectives

The main objectives are:

- Implement a DQN-based robot navigation system.
- Simulate a mobile robot in a 2D environment.
- Use simulated LiDAR measurements to detect obstacles.
- Train the robot to reach a target while avoiding collisions.
- Improve the state representation using sine and cosine.
- Compare DQN networks with different numbers of hidden layers.
- Evaluate navigation performance using experimental results.

## 3. Simulation Environment

The navigation environment is a two-dimensional workspace.

| Component | Description |
|---|---|
| Environment size | 300 × 300 units |
| Robot | Mobile robot |
| Obstacles | Two rectangular obstacles |
| Goal | Target region |
| Sensor | Simulated LiDAR |
| LiDAR rays | 61 |
| Sensor field of view | 180 degrees |
| Angular resolution | 3 degrees |
| Maximum episode length | 1,000 steps |

Each episode ends when:

1. The robot reaches the target.
2. The robot collides with an obstacle.
3. The robot reaches the maximum number of steps.

## 4. State Representation

The DQN receives a state vector containing 64 values.

The improved state representation includes:

- Normalized distance to the goal.
- Sine of the relative goal angle.
- Cosine of the relative goal angle.
- 61 normalized LiDAR distance measurements.

The state vector is:

```text
State = [
    normalized_goal_distance,
    sin(relative_goal_angle),
    cos(relative_goal_angle),
    lidar_1,
    lidar_2,
    ...
    lidar_61
]
```

Total input features:

```text
1 + 2 + 61 = 64
```

Using sine and cosine avoids the discontinuity between +180 and -180 degrees.

Distance normalization also helps keep input values on a more consistent scale.

## 5. Action Space

The robot has three discrete actions:

| Action | Description |
|---|---|
| 0 | Move forward |
| 1 | Rotate left |
| 2 | Rotate right |

The DQN predicts one Q-value for each action.

The robot selects the action with the highest Q-value during evaluation.

## 6. Reward Function

The reward function encourages the robot to reach its target while avoiding collisions and unnecessary movements.

| Situation | Reward |
|---|---|
| Reach the target | +100 |
| Collision | -100 |
| Intermediate movement | Based on progress toward the goal |

The intermediate reward is calculated as:

```text
reward = 2 × (previous_goal_distance - current_goal_distance) - 0.1
```

The progress term encourages the robot to move closer to the goal.

The step penalty discourages unnecessary movements.

## 7. Deep Q-Network Architecture

Three DQN architectures are investigated.

### DQN-1H

```text
Input Layer: 64 neurons
       |
Hidden Layer: 256 neurons (ReLU)
       |
Output Layer: 3 Q-values
```

Trainable parameters: **17,411**

### DQN-2H

```text
Input Layer: 64 neurons
       |
Hidden Layer 1: 256 neurons (ReLU)
       |
Hidden Layer 2: 256 neurons (ReLU)
       |
Output Layer: 3 Q-values
```

Trainable parameters: **83,203**

### DQN-3H

```text
Input Layer: 64 neurons
       |
Hidden Layer 1: 256 neurons (ReLU)
       |
Hidden Layer 2: 256 neurons (ReLU)
       |
Hidden Layer 3: 256 neurons (ReLU)
       |
Output Layer: 3 Q-values
```

Trainable parameters: **148,995**

All three architectures use fully connected neural networks.

The output layer produces three Q-values without a Softmax activation.

## 8. DQN Learning Algorithm

The DQN algorithm uses the following techniques:

### Experience Replay

The robot stores experiences in a replay buffer.

Each experience contains:

```text
(state, action, reward, next_state, done)
```

Random mini-batches are sampled from the replay buffer during training.

### Online Network

The Online Network predicts Q-values and is updated through backpropagation.

### Target Network

The Target Network calculates more stable target Q-values.

Its weights are periodically copied from the Online Network.

### Epsilon-Greedy

During training, the robot sometimes selects random actions to explore the environment.

As training progresses, epsilon decreases and the robot increasingly relies on learned Q-values.

### Bellman Target

For a non-terminal transition:

```text
target = reward + gamma × max(Q_target(next_state))
```

For a terminal transition:

```text
target = reward
```

The network minimizes the Mean Squared Error (MSE) between predicted Q-values and target Q-values.

Adam is used to update the network weights.

## 9. Training Configuration

| Parameter | Value |
|---|---|
| Algorithm | DQN |
| Framework | PyTorch |
| Training steps | 1,000,000 per model |
| Discount factor (gamma) | 0.99 |
| Replay buffer capacity | 30,000 |
| Batch size | 64 |
| Training frequency | Every 4 steps |
| Target network update | Every 10,000 steps |
| Initial epsilon | 1.0 |
| Final epsilon | 0.01 |
| Epsilon decay | 400,000 steps |
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| Training seed | 42 |

## 10. Experimental Results

All three models were evaluated using the same 300 navigation scenarios.

The evaluation seed was 2026, and epsilon was set to zero.

### Navigation Performance

| Metric | DQN-1H | DQN-2H | DQN-3H |
|---|---:|---:|---:|
| Success rate | 86.67% | 82.33% | 97.00% |
| Successful episodes | 260 | 247 | 291 |
| Collision episodes | 0 | 7 | 0 |
| Timeout episodes | 40 | 46 | 9 |
| Average successful steps | 288.57 | 308.51 | 272.09 |
| Average successful path length | 177.6 | 179.9 | 160.5 |
| Trainable parameters | 17,411 | 83,203 | 148,995 |

### Experimental Observations

- DQN-3H achieved the highest success rate at 97%.
- DQN-3H had the lowest timeout rate.
- DQN-3H required fewer steps and a shorter path on average in successful episodes.
- DQN-1H achieved good navigation performance with significantly fewer parameters.
- DQN-2H performed worse than DQN-1H in this experiment despite having more parameters.

These findings suggest that network depth can affect navigation performance, but adding layers does not always guarantee better results.

The results are based on a limited simulation environment and one training seed per architecture.

## 11. State Representation Experiments

The original state representation used raw relative goal angles.

This introduced a discontinuity around the +180/-180 degree boundary.

In some situations, the robot repeatedly switched between turning left and right.

The improved state uses sine and cosine to represent the relative goal direction.

Earlier experiments produced the following results:

| Configuration | Success Rate |
|---|---:|
| Original state, step penalty -0.02 | 50.0% |
| Original state, step penalty -0.1 | 60.5% |
| Improved Sin/Cos state, DQN-2H | 83.5% |

These results were obtained from earlier 200-episode evaluations.

The improved state also uses normalized distances, so the improvement cannot be attributed to sine and cosine alone.

## 12. Dataset and Training Data

This project does not use a fixed external dataset.

Training data is generated automatically through interactions between the robot and the simulation environment.

Each transition contains:

```text
Current State
Action
Reward
Next State
Done Flag
```

These transitions are stored in the Experience Replay Buffer and sampled during training.

The project also generates experimental data such as:

- Episode rewards.
- Discounted returns.
- Training losses.
- Episode lengths.
- Success rates.
- Collision rates.
- Timeout rates.
- Evaluation summaries.

Generated experimental results are stored in the `results/` directory.

## 13. Requirements

The project uses:

- Python 3.
- PyTorch.
- NumPy.
- Matplotlib.
- Tkinter.
- tqdm.
- more-itertools.

The program was developed and tested on Ubuntu.

A CPU can be used for training and evaluation.

## 14. Installation

Clone the repository:

```bash
git clone https://github.com/dungphung27/DRL_robot_navigation.git
cd DRL_robot_navigation
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Install Python packages:

```bash
pip install torch numpy matplotlib tqdm more-itertools
```

If Tkinter is not available on Ubuntu:

```bash
sudo apt install python3-tk
```

## 15. Running the Project

The main DQN implementation is located in:

```text
dqn.py
```

To run the DQN script:

```bash
python dqn.py
```

The exact training and evaluation options depend on the corresponding scripts and configurations.

Use the experiment scripts included in the repository to reproduce the different network architectures and compare their results.

The improved state representation can be enabled using:

```bash
export DQN_STATE_ENCODING=sincos
```

## 16. Testing

The repository includes several test scripts:

```text
test_episode.py
test_sensor.py
test_wall_sensor.py
test_easy_v2.py
trace_dqn_actions.py
```

Example:

```bash
python test_sensor.py
python test_episode.py
```

These scripts help inspect the simulation environment, sensors, episode behavior, and selected robot actions.

## 17. Experimental Outputs

The `results/` directory contains generated results from training and evaluation.

These may include:

- Training curves.
- TD loss graphs.
- Discounted return graphs.
- Success rate graphs.
- Collision and timeout graphs.
- Evaluation summaries.
- CSV files.
- PNG and PDF figures.

The figures and tables can be used for the final project report.

## 18. Project Limitations

The project currently has several limitations:

1. The robot is trained in a small 2D environment.
2. The experimental map contains two fixed rectangular obstacles.
3. Each architecture was trained using one random seed.
4. The robot has not been tested on real hardware.
5. The models have not been evaluated on completely new obstacle layouts.
6. Zero collisions in a limited evaluation does not guarantee zero collisions in future navigation.

## 19. Future Work

Possible improvements include:

- Training with multiple random seeds.
- Testing on more complex environments.
- Using different obstacle configurations.
- Investigating remaining timeout cases.
- Comparing other reinforcement learning algorithms.
- Testing Double DQN.
- Evaluating the system on a physical mobile robot.

## 20. Author

**GitHub:** dungphung27

**Project Type:** Individual Deep Learning Course Project

**Repository:**
https://github.com/dungphung27/DRL_robot_navigation

## 21. Conclusion

This project demonstrates how Deep Q-Network can be applied to autonomous robot navigation.

The experiments show that state representation and neural network architecture both affect navigation performance.

Among the tested models, DQN-3H achieved the highest success rate of 97% on 300 evaluation episodes.

DQN-1H remains a useful alternative when a smaller network is preferred.

The project provides a foundation for further research into reinforcement learning for autonomous robots.
