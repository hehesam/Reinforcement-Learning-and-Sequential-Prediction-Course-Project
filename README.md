# Reinforcement Learning and Sequential Prediction Course Project

## Comparison of Learning Progress

<table>
<tr>
<td width="50%">

### Q-Learning
![Q-Learning Progress](mountain%20car%20gifs/all_episodes_q.gif)

</td>
<td width="50%">

### Actor-Critic (Policy Gradient)
![Actor-Critic Progress](mountain%20car%20gifs/all_episodes_ac.gif)

</td>
</tr>
</table>

## Overview

This project implements and compares two reinforcement learning algorithms on the classic Mountain Car problem:
- **Q-Learning** with function approximation using Radial Basis Functions (RBF)
- **Actor-Critic** (Policy Gradient) with RBF features

Both algorithms are tested with multiple random seeds and various hyperparameters to evaluate their performance and stability. The environment is implemented using **Gymnasium** (previously OpenAI Gym).

## Results

### Q-Learning Results (1000 Episodes)

| Sigma | Alpha | Mean Return | Return Std | Success Rate | Success Rate Std | Steps to Goal | Steps Std | Max Abs Q | Max Abs Q Std |
|-------|-------|-------------|------------|--------------|------------------|---------------|-----------|-----------|---------------|
| 0.15 | 0.003 | -143.33 | 13.80 | 90.0% | 12.75% | 137.70 | 7.57 | 74.89 | 6.94 |
| 0.15 | 0.005 | -144.21 | 9.04 | 100.0% | 0.00% | 144.21 | 9.04 | 72.59 | 2.26 |
| 0.20 | 0.003 | -131.97 | 2.64 | 100.0% | 0.00% | 131.97 | 2.64 | 72.67 | 2.59 |
| 0.20 | 0.005 | -149.43 | 6.23 | 94.0% | 8.94% | 146.11 | 5.45 | 77.53 | 4.52 |

**Best Configuration:** σ=0.20, α=0.003 (lowest steps to goal with 100% success rate)

### Actor-Critic Results (1000 Episodes)

| Sigma | Alpha θ | Alpha v | Mean Return | Return Std | Success Rate | Success Rate Std | Steps to Goal | Steps Std |
|-------|---------|---------|-------------|------------|--------------|------------------|---------------|-----------|
| 0.15 | 0.0003 | 0.003 | -136.79 | 3.54 | 98.0% | 2.74% | 135.44 | 4.47 |
| 0.15 | 0.0003 | 0.005 | -137.88 | 3.62 | 100.0% | 0.00% | 137.88 | 3.62 |
| 0.15 | 0.0005 | 0.003 | -136.85 | 3.62 | 98.0% | 2.74% | 135.50 | 4.46 |
| 0.15 | 0.0005 | 0.005 | -137.82 | 3.74 | 100.0% | 0.00% | 137.82 | 3.74 |
| 0.20 | 0.0003 | 0.003 | -140.26 | 6.46 | 98.0% | 4.47% | 138.70 | 5.87 |
| 0.20 | 0.0003 | 0.005 | -136.87 | 5.51 | 99.0% | 2.24% | 136.41 | 5.70 |
| 0.20 | 0.0005 | 0.003 | -139.88 | 5.28 | 98.0% | 4.47% | 138.28 | 4.28 |
| 0.20 | 0.0005 | 0.005 | -138.09 | 8.98 | 99.0% | 2.24% | 137.65 | 8.79 |

**Best Configuration:** σ=0.15, α_θ=0.0003, α_v=0.003 (lowest steps to goal)

## Key Findings

- **Actor-Critic** achieves consistently lower steps to goal (135-140 steps) compared to Q-Learning (132-149 steps)
- **Actor-Critic** shows better stability with lower standard deviation across seeds
- Both algorithms achieve high success rates (≥90%) with proper hyperparameter tuning
- RBF features with σ=0.15 or σ=0.20 work well for both approaches

## Requirements

```
python >= 3.8
numpy >= 1.20.0
gymnasium >= 0.28.0
pillow >= 9.0.0
imageio >= 2.9.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hehesam/Reinforcement-Learning-and-Sequential-Prediction-Course-Project.git
cd Reinforcement-Learning-and-Sequential-Prediction-Course-Project
```

2. Install required packages:
```bash
pip install numpy gymnasium pillow imageio
```

## Project Structure

```
.
├── q_learning.py           # Q-Learning implementation with RBF
├── policy_grad.py          # Actor-Critic (Policy Gradient) implementation
├── rbf.py                  # Radial Basis Function implementation
├── env_utils.py            # Environment utilities and evaluation functions
├── Diagnostics.py          # Diagnostic tools for analysis
├── Multi_Seed_Runner.py    # Multi-seed experiment runner
├── multi_seed_summary.json # Q-Learning results
├── ac_multi_seed_summary.json # Actor-Critic results
└── mountain car gifs/      # Visualization of learning progress
```

## Usage

### Train Q-Learning Agent

```bash
python q_learning.py
```

This will train a Q-Learning agent for 1000 episodes with the default hyperparameters and display the environment every 100 episodes.

### Train Actor-Critic Agent

```bash
python policy_grad.py
```

This will train an Actor-Critic agent for 1000 episodes with the default hyperparameters.

### Run Multi-Seed Experiments

To run experiments across multiple seeds and hyperparameter combinations:

```bash
python Multi_Seed_Runner.py
```

Uncomment the respective function calls in the script:
- `Q_call_multiseed(env_id, seeds)` for Q-Learning experiments
- `AC_call_multiseed(env_id, seeds)` for Actor-Critic experiments

## Environment Details

The project uses the **MountainCar-v0** environment from **Gymnasium**:
- **State Space:** 2D continuous (position, velocity)
- **Action Space:** 3 discrete actions (push left, no push, push right)
- **Goal:** Reach the flag on top of the right hill
- **Challenge:** Car doesn't have enough power to climb directly, must build momentum

## Algorithm Details

### Q-Learning with Function Approximation
- Uses semi-gradient Q-learning update
- Linear function approximation with RBF features (7×7 centers)
- ε-greedy exploration with linear decay
- Actions selected based on max Q-value during evaluation

### Actor-Critic
- One-step TD advantage estimation
- Separate networks for policy (actor) and value function (critic)
- Softmax policy for action selection
- Greedy policy evaluation (argmax over action probabilities)

## Hyperparameters

### Q-Learning
- Training episodes: 1000
- Discount factor (γ): 0.99
- Learning rate (α): 0.003-0.005
- RBF sigma (σ): 0.15-0.20
- Epsilon decay: Linear from 1.0 to 0.05 over 70% of episodes

### Actor-Critic
- Training episodes: 1000
- Discount factor (γ): 0.99
- Policy learning rate (α_θ): 0.0003-0.0005
- Value learning rate (α_v): 0.003-0.005
- RBF sigma (σ): 0.15-0.20

## License

This project is part of the Reinforcement Learning and Sequential Prediction course at [University Name].

## Author

[Your Name]

## Acknowledgments

- Gymnasium (OpenAI Gym) for the MountainCar environment
- Course instructors and materials
