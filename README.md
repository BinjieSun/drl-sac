# Soft Actor-Critic (SAC) Implementation in PyTorch

This is a PyTorch implementation of Soft Actor-Critic (SAC), an off-policy model-free reinforcement learning algorithm. SAC combines the benefits of off-policy learning with a maximum entropy framework to encourage exploration while maintaining sample efficiency.

Original paper: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290), [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905)

## Modifications of the original implementation
- Implementation of SAC in PyTorch based on pytorch_sac from denisyarats
- Added support for Gymnasium environment
- Revamped the support for DMC in Gymnasium
- Revamped the support for built-in video feature in Gymnasium
- Updated the usages of packages

## Setup
### Quick experiment
Quick way to experiment with the SAC implementation:
- Run the experiment.ipynb file in Google Colab
- Follow through the steps to clone the repo and install packages
- Experiment with different environment

### Experiment on your own GPU
Train an SAC agent on your own GPU
- Download the file
- Install packages
- Start Training

### Configurations
Change configurations in the config folder to test out different parameters. To train the model, use
```bash
python train.py
```

0r to specify a different environment:

```bash
python train.py env_type=gym env=Hopper-v5
```

Available environments include:

Gymnasium (a maintained fork of OpenAI’s Gym library)
- Hopper-v5
- Walker2d-v5
- HalfCheetah-v5
- and more

DMC (Deep Mind Control Suite)
- cheetah_run
- walker_walk
- ball_in_cup_catch
- and more

## Results
The SAC implementation has been tested on . The results show competitive performance across various tasks with fixed hyperparameters.

add result image here