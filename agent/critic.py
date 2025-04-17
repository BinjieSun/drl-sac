import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import Linear

import utils

from agent.mygnn import MyGNN

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, trunk_type='mlp', env_name='ant'):
        super().__init__()

        if trunk_type == 'mlp':
            self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
            self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        elif trunk_type == 'gnn':
            self.Q1 = MyGNN(env_name.lower(), hidden_dim, is_critic=True)
            self.Q2 = MyGNN(env_name.lower(), hidden_dim, is_critic=True)
        else:
            raise ValueError(f"Invalid trunk type: {trunk_type}")

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)
                
        if isinstance(self.Q1, nn.Sequential) and isinstance(self.Q2, nn.Sequential):
            # Original code for nn.Sequential
            for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
                assert type(m1) == type(m2)
                if type(m1) is nn.Linear:
                    logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                    logger.log_param(f'train_critic/q2_fc{i}', m2, step)
        else:
            # New code for handling MyGNN objects
            # Log Q1 parameters
            # Log parameters in node_feature_extractor
            for node_id, mlp in self.Q1.node_feature_extractor.node_mlps.items():
                for i, layer in enumerate(mlp):
                    if isinstance(layer, nn.Linear):
                        logger.log_param(f'train_critic/q1_node_{node_id}_mlp_{i}', layer, step)
            
            # Log convolution layer parameters
            for i, conv in enumerate(self.Q1.convs):
                if hasattr(conv, 'lin'):
                    logger.log_param(f'train_critic/q1_conv_{i}_lin', conv.lin, step)
                elif hasattr(conv, 'lin_l'):
                    logger.log_param(f'train_critic/q1_conv_{i}_lin_l', conv.lin_l, step)
                elif hasattr(conv, 'lin_r'):
                    logger.log_param(f'train_critic/q1_conv_{i}_lin_r', conv.lin_r, step)
            
            # Log decoder parameters
            for i, layer in enumerate(self.Q1.decoder):
                if isinstance(layer, nn.Linear) or isinstance(layer, Linear):
                    logger.log_param(f'train_critic/q1_decoder_{i}', layer, step)
            
            # Log Q2 parameters
            # Log parameters in node_feature_extractor
            for node_id, mlp in self.Q2.node_feature_extractor.node_mlps.items():
                for i, layer in enumerate(mlp):
                    if isinstance(layer, nn.Linear):
                        logger.log_param(f'train_critic/q2_node_{node_id}_mlp_{i}', layer, step)
            
            # Log convolution layer parameters
            for i, conv in enumerate(self.Q2.convs):
                if hasattr(conv, 'lin'):
                    logger.log_param(f'train_critic/q2_conv_{i}_lin', conv.lin, step)
                elif hasattr(conv, 'lin_l'):
                    logger.log_param(f'train_critic/q2_conv_{i}_lin_l', conv.lin_l, step)
                elif hasattr(conv, 'lin_r'):
                    logger.log_param(f'train_critic/q2_conv_{i}_lin_r', conv.lin_r, step)
            
            # Log decoder parameters
            for i, layer in enumerate(self.Q2.decoder):
                if isinstance(layer, nn.Linear) or isinstance(layer, Linear):
                    logger.log_param(f'train_critic/q2_decoder_{i}', layer, step)
