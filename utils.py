import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import gymnasium as gym
import os
from collections import deque
import random
import math


def make_env(cfg):
    """Helper function to create environment"""
    if cfg.env_type == 'dmc':
        return make_dmc_env(cfg)
    elif cfg.env_type == 'gym':
        return make_gym_env(cfg)
    else:
        raise ValueError(f"Unsupported environment type: {cfg.env_type}")

def make_dmc_env(cfg, for_evaluation=False):
    """Helper function to create dm_control environment"""
    try:
        from dm_control import suite
        from dm_env import specs
        import dm_env
    except ImportError:
        raise ImportError("DMC environments require dm_control. Install with: pip install dm_control")
        
    # Parse domain and task from cfg.env (format: domain_task)
    if '_' not in cfg.env:
        raise ValueError(f"DMC env should be in format 'domain_task', got: {cfg.env}")
        
    domain, task = cfg.env.split('_', 1)
    
    # Set MUJOCO_GL for headless rendering if needed
    os.environ['MUJOCO_GL'] = 'egl'
    
    # Load DMC environment
    dmc_env = suite.load(domain, task, task_kwargs={'random': cfg.seed})
    
    # Wrap DMC environment in Gymnasium adapter
    env = DMCGymWrapper(dmc_env)
    
    return env

def make_gym_env(cfg, for_evaluation=False):
    """Helper function to create OpenAI Gym environment"""
    # Set appropriate GL backend for headless rendering
    os.environ['MUJOCO_GL'] = 'egl'  # Try 'egl' first as it's often faster than osmesa
    
    # Choose render mode based on whether this is for evaluation and if video saving is enabled
    # If not evaluating or not saving video, use None to avoid rendering issues
    if for_evaluation and cfg.get('save_video', False):
        render_mode = 'rgb_array_list'  # Use rgb_array_list for video recording
    elif for_evaluation:
        render_mode = 'rgb_array'  # Use rgb_array for evaluation visualization
    else:
        render_mode = None  # Don't render during training - important for headless environments
    
    # Create env without seed in make() for MuJoCo envs
    env = gym.make(cfg.env, render_mode=render_mode)
    
    # Check if action space needs normalization
    if isinstance(env.action_space, gym.spaces.Box):
        # Check if action space is not already in [-1, 1]
        if (env.action_space.low != -1).any() or (env.action_space.high != 1).any():
            print(f"Warning: Environment {cfg.env} has action space with range " 
                  f"[{env.action_space.low.min()}, {env.action_space.high.max()}]. "
                  f"Actions will be scaled during agent interaction.")
            
            # Create a wrapper to normalize actions
            env = NormalizedActionWrapper(env)
    
    return env

class DMCGymWrapper(gym.Env):
    """Wrapper to convert DeepMind Control Suite environments to Gymnasium interface."""
    
    def __init__(self, env):
        self.env = env
        self.metadata = {'render.modes': ['rgb_array']}
        
        # Get action spec and observation spec
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()
        
        # Define action space (always normalized to [-1, 1] in DMC)
        self.action_space = gym.spaces.Box(
            low=np.full(self.action_spec.shape, -1.0, dtype=np.float32),
            high=np.full(self.action_spec.shape, 1.0, dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space
        obs_spaces = {}
        for key, spec in self.observation_spec.items():
            if isinstance(spec.shape, (list, tuple)):
                obs_spaces[key] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=spec.shape, dtype=np.float32
                )
            else:
                obs_spaces[key] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(spec.shape,), dtype=np.float32
                )
        
        self.observation_space = gym.spaces.Dict(obs_spaces)
        
    def _get_obs(self, time_step):
        """Extract observation from dm_env TimeStep."""
        return {k: np.asarray(v, dtype=np.float32) for k, v in time_step.observation.items()}
    
    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        time_step = self.env.reset()
        obs = self._get_obs(time_step)
        info = {}
        return obs, info
    
    def step(self, action):
        """Take a step in environment."""
        action = np.asarray(action, dtype=np.float32)
        time_step = self.env.step(action)
        
        obs = self._get_obs(time_step)
        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False  # DMC doesn't support truncation
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render environment."""
        from dm_control.viewer import renderer
        return renderer.OffScreenRenderer(self.env.physics.model.ptr, *self.env.physics.data.ptr).render()

class NormalizedActionWrapper(gym.ActionWrapper):
    """Wrapper to normalize action spaces to [-1, 1]."""
    def __init__(self, env):
        super(NormalizedActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype
        )
    
    def action(self, action):
        # Scale from [-1, 1] to original action space
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        return action
    
    # In Gymnasium, reverse_action is not used anymore
    # Instead, they now use "unwrap_action" if needed
    def unwrap_action(self, action):
        # Scale from original action space to [-1, 1]
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = 2.0 * (action - low) / (high - low) - 1.0
        action = np.clip(action, -1.0, 1.0)
        return action


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
