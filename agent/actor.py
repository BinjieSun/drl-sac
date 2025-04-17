import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch_geometric.nn import Linear
import utils
from agent.mygnn import MyGNN

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        # self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
        #                        hidden_depth)
        
        self.trunk = MyGNN('humanoid-v5', hidden_dim)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        if isinstance(self.trunk, nn.Sequential):
            # Original code for nn.Sequential
            for i, m in enumerate(self.trunk):
                if type(m) == nn.Linear:
                    logger.log_param(f'train_actor/fc{i}', m, step)
        else:
            # New code for handling MyGNN objects
            # Log parameters in node_feature_extractor
            for node_id, mlp in self.trunk.node_feature_extractor.node_mlps.items():
                for i, layer in enumerate(mlp):
                    if isinstance(layer, nn.Linear):
                        logger.log_param(f'train_actor/node_{node_id}_mlp_{i}', layer, step)
            
            # Log convolution layer parameters
            for i, conv in enumerate(self.trunk.convs):
                if hasattr(conv, 'lin'):
                    logger.log_param(f'train_actor/conv_{i}_lin', conv.lin, step)
                elif hasattr(conv, 'lin_l'):
                    logger.log_param(f'train_actor/conv_{i}_lin_l', conv.lin_l, step)
                elif hasattr(conv, 'lin_r'):
                    logger.log_param(f'train_actor/conv_{i}_lin_r', conv.lin_r, step)
            
            # Log decoder parameters
            for i, layer in enumerate(self.trunk.decoder):
                if isinstance(layer, nn.Linear) or isinstance(layer, Linear):
                    logger.log_param(f'train_actor/decoder_{i}', layer, step)
