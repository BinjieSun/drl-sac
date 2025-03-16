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
- Open `experiment.ipynb` in Google Colab
- Follow the steps to clone the repository and install dependencies
- Train an SAC agent

### Running locally on your GPU
- Clone the repository
- Install dependencies: `pip install -r requirements.txt`
- Train an SAC agent

### Configurations
To train the model, use:
```bash
python train.py
```

0r specify a different environment:

```bash
python train.py env_type=gym env=Hopper-v5
```
Modify configurations in the `config` folder to experiment with different model parameters. 

Available environments:
- Gymnasium: Hopper, Walker2d, HalfCheetah, and [more](https://gymnasium.farama.org/environments/mujoco/).
- Deep Mind Control Suite: cheetah_run, walker_walk, ball_in_cup_catch, and [more](https://github.com/google-deepmind/dm_control/tree/main/dm_control/suite).

## Results
We tested the SAC implementation on Hopper-v5, Walker2d-v5, HalfCheetah-v5, Ant-v5, and Humanoid-v5. The results aligns closely with those reported in the original SAC paper.

*Note: add result image here*
