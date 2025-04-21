# Soft Actor-Critic (SAC) Implementation in PyTorch

This is a PyTorch implementation of Soft Actor-Critic (SAC), an off-policy, model-free reinforcement learning algorithm. SAC leverages a maximum entropy framework to encourage exploration while maintaining sample efficiency.
- Original paper: [Soft Actor-Critic](https://arxiv.org/pdf/1801.01290), [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905)
- Original implementation: [pytorch_sac](https://github.com/denisyarats/pytorch_sac) from denisyarats

## Modifications to the original implementation
This SAC implementation extends the original one by introducing:
- Support for Gymnasium environments
- Built-in video recording in Gymnasium
- A DeepMind Control Suite (DMC) wrapper in Gymnasium
- Added model checkpoints to save the model and prevent data loss

## Setup
### Running an experiment in Colab
- Open `experiment.ipynb` to install dependencies
- Train an SAC agent using the different scripts
```
python train_walker2d.py
python train_humanoid.py
```

The scripts could be found in both `swei-dev` branch and `swei-hopper` branch.

Modify configurations in the `config` folder to experiment with different model parameters. 

Available environments:
- Gymnasium: Hopper, Walker2d, HalfCheetah, and [more](https://gymnasium.farama.org/environments/mujoco/).
- Deep Mind Control Suite: cheetah_run, walker_walk, ball_in_cup_catch, and [more](https://github.com/google-deepmind/dm_control/tree/main/dm_control/suite).

## Results
We tested the SAC implementation on Hopper-v5, Walker2d-v5, HalfCheetah-v5, Ant-v5, and Humanoid-v5. The results aligns with those reported in the original SAC++ paper.
[Video demo](https://www.youtube.com/watch?v=4yIPq6WdDSI)
![fig1](https://github.com/user-attachments/assets/c6d354a8-58db-4023-87a7-5fc268798a8b)
![fig2](https://github.com/user-attachments/assets/22cd0098-ad47-479d-93f1-0e5b4613727a)

