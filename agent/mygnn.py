import numpy as np

import torch
import torch.nn as nn

from torch_geometric.nn import Linear, GraphConv
from copy import deepcopy

class MyGNN(torch.nn.Module):
    """
    Standard GNN
    """
    def __init__(self, task, hidden_channels: int, num_layers: int, 
                activation_fn = nn.ELU(), batch_size: int = 1024, is_critic: bool = False):
        """
        Implementation of a standard GNN model for C2 structure.

        Parameters:
            hidden_channels (int): Size of the node embeddings in the graph.
            num_layers (int): Number of message-passing layers.
            activation_fn (class): The activation function used between layers.
        """
        super().__init__()
        self.activation = activation_fn
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Total number of nodes (combining base and joint)
        self.num_base_nodes = 1
        self.num_joint_nodes = 12
        self.num_nodes = self.num_base_nodes + self.num_joint_nodes
        
        # Create a single edge_index for all connections
        self.edge_index = self._create_edges_index()
        
        self.is_critic = is_critic
        
        if 'push_door' in task:
            self.num_timesteps = 5
            self.common_obs_base_indices = [*range(0, 6), *range(30, 43)]
            self.common_obs_joint_indices = [*range(6, 30)]
            if self.is_critic:
                self.privileged_obs_base_indices = [*range(8), *range(20, 35)]
                self.privileged_obs_joint_indices = [*range(8, 20)]
        else:
            self.num_timesteps = 3
            self.common_obs_base_indices = [*range(9), *range(45, 47)]
            self.common_obs_joint_indices = [*range(9, 45)]
            if self.is_critic:
                self.privileged_obs_base_indices = [*range(8), *range(20, 35)]
                self.privileged_obs_joint_indices = [*range(8, 20)]
        
        self.len_common_obs = self.num_timesteps * (len(self.common_obs_base_indices) + len(self.common_obs_joint_indices))
        
        # Calculate input feature dimensions for each node type
        self.base_feature_dim = len(self.common_obs_base_indices) * self.num_timesteps
        self.joint_feature_dim = (len(self.common_obs_joint_indices) // self.num_joint_nodes) * self.num_timesteps
        
        if self.is_critic:
            self.base_feature_dim += len(self.privileged_obs_base_indices)
            self.joint_feature_dim += len(self.privileged_obs_joint_indices) // self.num_joint_nodes
        
        # Create separate encoders for base and joint nodes
        self.base_encoder = nn.Linear(self.base_feature_dim, hidden_channels)
        self.joint_encoder = nn.Linear(self.joint_feature_dim, hidden_channels)

        # Create standard graph convolutions for each layer
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # Use standard GraphConv for all edges
            self.convs.append(GraphConv(hidden_channels, hidden_channels))

        # Output layer
        self.out_channels_per_node = 1
        
        if self.is_critic:
            self.decoder = nn.Sequential(
                Linear(hidden_channels * self.num_nodes, hidden_channels),
                self.activation,
                Linear(hidden_channels, 1)
            )
        else:
            self.decoder = Linear(hidden_channels, self.out_channels_per_node)

        # Create batched edge indices for common batch sizes
        self.edge_index_batch_1024 = self._create_edge_index_batch(1024).to(self.device)
        self.edge_index_batch_6144 = self._create_edge_index_batch(6144).to(self.device)

    def _create_edge_index_batch(self, batch_size):
        '''
        Create edge index for batch size
        '''
        edge_indices = []
        for i in range(batch_size):
            # Add edges with batch-specific offsets
            node_offset = i * self.num_nodes
            edge_offset = torch.ones(2, self.edge_index.shape[1], dtype=torch.int64) * node_offset
            edge_indices.append(self.edge_index + edge_offset)
        
        # Concatenate all edge indices
        edge_index_batch = torch.cat(edge_indices, dim=1).to(self.device)
        return edge_index_batch

    def _create_edges_index(self):
        # Define all connections in a single edge_index tensor        
        node_2_node = torch.tensor([[0, 0, 0, 0, 1, 2, 4, 5, 7, 8, 10, 11],
                                    [1, 4, 7, 10, 2, 3, 5, 6, 8, 9, 11, 12]])

        edge_index = torch.cat([
            node_2_node, node_2_node.flip(0)
        ], dim=1)
        
        return edge_index
    
    def _obs_to_graph_features(self, obs_all):
        """
        Convert observations into features for each node in a batch of graphs.
        """
        batch_size = obs_all.shape[0]
        obs = deepcopy(obs_all[:, :self.len_common_obs])
        obs = obs.reshape(batch_size, self.num_timesteps, -1)

        # Extract base features
        base_feature = obs[:, :, self.common_obs_base_indices].reshape(batch_size, self.num_base_nodes, self.num_timesteps, len(self.common_obs_base_indices))
        base_feature = base_feature.reshape(batch_size, self.num_base_nodes, -1)  # [batch_size, 1, features]

        # Extract joint features
        joint_feature = obs[:, :, self.common_obs_joint_indices].reshape(batch_size, self.num_timesteps, len(self.common_obs_joint_indices)//self.num_joint_nodes, self.num_joint_nodes).permute(0, 3, 1, 2)
        joint_feature = joint_feature.reshape(batch_size, self.num_joint_nodes, -1)  # [batch_size, 12, features]
        
        if self.is_critic:
            privileged_obs = deepcopy(obs_all[:, self.len_common_obs:])
            privileged_base_feature = privileged_obs[:, self.privileged_obs_base_indices].reshape(batch_size, 1, -1)
            privileged_joint_feature = privileged_obs[:, self.privileged_obs_joint_indices].reshape(batch_size, -1, self.num_joint_nodes).permute(0, 2, 1)
            base_feature = torch.cat((base_feature, privileged_base_feature), dim=-1)
            joint_feature = torch.cat((joint_feature, privileged_joint_feature), dim=-1)

        # Reshape for processing
        base_feature = base_feature.reshape(batch_size * self.num_base_nodes, -1)  # [batch_size * 2, features]
        joint_feature = joint_feature.reshape(batch_size * self.num_joint_nodes, -1)  # [batch_size * 12, features]

        if batch_size == 1024:
            edge_index = self.edge_index_batch_1024
        elif batch_size == 6144:
            edge_index = self.edge_index_batch_6144
        else:
            print(f"-------Unknown batch size: {batch_size}-------")
            edge_index = self._create_edge_index_batch(batch_size).to(self.device)
            
        return base_feature, joint_feature, edge_index

    def forward(self, obs):
        batch_size = obs.shape[0]

        base_feature, joint_feature, edge_index = self._obs_to_graph_features(obs)

        # Initial feature encoding - separate for base and joint nodes
        base_encoded = self.activation(self.base_encoder(base_feature)).reshape(batch_size, self.num_base_nodes, -1)  # [batch_size, 1, hidden_channels]
        joint_encoded = self.activation(self.joint_encoder(joint_feature)).reshape(batch_size, self.num_joint_nodes, -1)  # [batch_size, 12, hidden_channels]
        
        x = torch.cat((base_encoded, joint_encoded), dim=1).reshape(batch_size * self.num_nodes, -1)  # [batch_size * 13, hidden_channels]
        
        # Message passing layers
        for conv in self.convs:
            # Apply convolution
            x_new = conv(x, edge_index)
            x_new = self.activation(x_new)
            x = x_new
        
        if self.is_critic:
            # For critic, use all node embeddings
            x_reshaped = x.reshape(batch_size, -1)  # [batch_size, num_nodes * hidden_channels]
            final_output = self.decoder(x_reshaped)  # [batch_size, 1]
        else:
            # For actor, use only joint node embeddings to produce actions
            # Extract only joint nodes (indices 2-13)
            joint_indices = torch.arange(self.num_base_nodes, self.num_nodes).to(self.device)
            joint_indices = joint_indices.repeat(batch_size) + torch.arange(0, batch_size * self.num_nodes, self.num_nodes).to(self.device).repeat_interleave(self.num_joint_nodes)
            joint_x = x[joint_indices]  # [batch_size * 12, hidden_channels]
            
            final_output = self.decoder(joint_x).reshape(batch_size, 12)  # [batch_size, 12]

        return final_output
    
    def reset_parameters(self):
        """Reset all learnable parameters"""
        self.encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if isinstance(self.decoder, nn.Sequential):
            for layer in self.decoder:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        else:
            self.decoder.reset_parameters()
