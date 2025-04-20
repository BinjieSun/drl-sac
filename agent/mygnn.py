import numpy as np

import torch
import torch.nn as nn

from torch_geometric.nn import Linear, GraphConv, LayerNorm
from copy import deepcopy

class NodeTypeSpecificMLP(nn.Module):
    def __init__(self, nodes_dict, hidden_channels, activation_fn):
        super(NodeTypeSpecificMLP, self).__init__()
        # Create MLPs for each node type
        self.node_mlps = nn.ModuleDict()
        self.hidden_channels = hidden_channels
        
        for node_id, node_info in nodes_dict.items():
            input_dim = len(node_info['feature_indices'])
            self.node_mlps[str(node_id)] = nn.Sequential(
                nn.Linear(input_dim, hidden_channels),
                activation_fn,
                nn.Linear(hidden_channels, hidden_channels)
            )
    
    def forward(self, x_feature_dict):
        """
        x_feature_dict: node feature dictionary
        """
        output = []
        
        for node_id, mlp in self.node_mlps.items():
            # Extract relevant features and pass them through the corresponding MLP
            node_features = x_feature_dict[int(node_id)]
            node_output = mlp(node_features)
            output.append(node_output.unsqueeze(1))
        
        output = torch.cat(output, dim=1).reshape(-1, self.hidden_channels)            
        return output # [batch_size * num_nodes, hidden_channels]

class MyGNN(torch.nn.Module):
    """
    Standard GNN
    """
    def __init__(self, task, hidden_channels: int = 128, num_layers: int = 8, 
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
        
        self.batch_size = batch_size
        self.is_critic = is_critic
        self.task = task
        if 'humanoid' in task:
            '''
            348 features
            | Joint Name      | Joint Index | *qpos & *qvel Feature Index          | qfrc_actuator Index |
            |-----------------|-------------|--------------------------------------|---------------------|
            | torso           | 0           | *range(0, 5), *range(22, 28)         |                     |
            | lwaist          | 1           | *range(5, 7), *range(28, 30)         |                     |
            | pelvis          | 2           | 7, 30                                | 0, 1, 2             |
            | right_thigh     | 3           | *range(8, 12), *range(31, 35)        | 3, 4, 5             |
            | right_shin      | 4           | 11, 34                               | 6                   |
            | right_foot      | 5           |                                      |                     |
            | left_thigh      | 6           | *range(12, 16), *range(35, 39)       | 7, 8, 9             |
            | left_shin       | 7           | 15, 38                               | 10                  |
            | left_foot       | 8           |                                      |                     |
            | right_upper_arm | 9           | *range(16, 19), *range(39, 42)       | 11, 12              |
            | right_lower_arm | 10          | 18, 41                               | 13                  |
            | left_upper_arm  | 11          | *range(19, 22), *range(42, 45)       | 14, 15              |
            | left_lower_arm  | 12          | 21, 44                               | 16                  |

            - *cinert (130 elements):* Mass and inertia of the rigid body parts relative to the center of mass,
            (this is an intermediate result of the transition).
            It has shape 13*10 (*nbody * 10*).
            (cinert - inertia matrix and body mass offset and body mass)
            - *cvel (78 elements):* Center of mass based velocity.
            It has shape 13 * 6 (*nbody * 6*).
            (com velocity - velocity x, y, z and angular velocity x, y, z)
            - *qfrc_actuator (17 elements):* Constraint force generated as the actuator force at each joint.
            This has shape `(17,)`  *(nv * 1)*.
            - *cfrc_ext (78 elements):* This is the center of mass based external force on the body parts.
            It has shape 13 * 6 (*nbody * 6*) and thus adds another 78 elements to the observation space.
            (external forces - force x, y, z and torque x, y, z)

            '''
            # Total number of nodes (combining base and joint)
            self.num_nodes = 13
            self.nodes_dict = {
                0: {'name': 'torso', 'feature_indices': [*range(0, 5), *range(22, 28)], 'qfrc_actuator_indices': []},
                1: {'name': 'lwaist', 'feature_indices': [*range(5, 7), *range(28, 30)], 'qfrc_actuator_indices': []},
                2: {'name': 'pelvis', 'feature_indices': [7, 30], 'qfrc_actuator_indices': [0, 1, 2]},
                3: {'name': 'right_thigh', 'feature_indices': [*range(8, 12), *range(31, 35)], 'qfrc_actuator_indices': [3, 4, 5]},
                4: {'name': 'right_shin', 'feature_indices': [11, 34], 'qfrc_actuator_indices': [6]},
                5: {'name': 'right_foot', 'feature_indices': [], 'qfrc_actuator_indices': []},
                6: {'name': 'left_thigh', 'feature_indices': [*range(12, 16), *range(35, 39)], 'qfrc_actuator_indices': [7, 8, 9]},
                7: {'name': 'left_shin', 'feature_indices': [15, 38], 'qfrc_actuator_indices': [10]},
                8: {'name': 'left_foot', 'feature_indices': [], 'qfrc_actuator_indices': []},
                9: {'name': 'right_upper_arm', 'feature_indices': [*range(16, 19), *range(39, 42)], 'qfrc_actuator_indices': [11, 12]},
                10: {'name': 'right_lower_arm', 'feature_indices': [18, 41], 'qfrc_actuator_indices': [13]},
                11: {'name': 'left_upper_arm', 'feature_indices': [*range(19, 22), *range(42, 45)], 'qfrc_actuator_indices': [14, 15]},
                12: {'name': 'left_lower_arm', 'feature_indices': [21, 44], 'qfrc_actuator_indices': [16]}
            }
            for key, value in self.nodes_dict.items():
                # cinert
                self.nodes_dict[key]['feature_indices'].extend([*range(45 + key * 10, 45 + (key + 1) * 10)])
                # cvel
                self.nodes_dict[key]['feature_indices'].extend([*range(175 + key * 6, 175 + (key + 1) * 6)])
                # qfrc_actuator
                if len(self.nodes_dict[key]['qfrc_actuator_indices']) > 0:
                    self.nodes_dict[key]['feature_indices'].extend([253 + idx for idx in self.nodes_dict[key]['qfrc_actuator_indices']])
                # cfrc_ext
                self.nodes_dict[key]['feature_indices'].extend([*range(270 + key * 6, 270 + (key + 1) * 6)])
                
                if self.is_critic:
                    # current joint actions
                    if len(self.nodes_dict[key]['qfrc_actuator_indices']) > 0:
                        self.nodes_dict[key]['feature_indices'].extend([348 + idx for idx in self.nodes_dict[key]['qfrc_actuator_indices']])
                
            node_2_node = torch.tensor([[0, 1, 2, 3, 4, 2, 6, 7, 0, 9, 0, 11],
                                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

            self.edge_index = torch.cat([
                node_2_node, node_2_node.flip(0)
            ], dim=1)
            
            self.num_joints = 17

        elif 'hopper' in task:
            """
            Hopper-v5
                torso
                  |
                0 thigh
                  |
                1 leg
                  |
                2 foot

            3 nodes, 11 features
            | Node Name | Node Index  | Feature Index        | Actuator Index |
            |-----------|-------------|----------------------|----------------|
            | thigh     | 0           | 0, 1, 5, 6, 7, 2, 8  | 11             |
            | shin      | 1           | 0, 1, 5, 6, 7, 3, 9  | 12             |
            | foot      | 2           | 0, 1, 5, 6, 7, 4, 10 | 13             |
            """
            self.num_nodes = 3
            self.num_joints = 3
            self.nodes_dict = {
                0: {'name': 'torso', 'feature_indices': [0, 1, 5, 6, 7, 2, 8], 'qfrc_actuator_indices': [11]},
                1: {'name': 'thigh', 'feature_indices': [0, 1, 5, 6, 7, 3, 9], 'qfrc_actuator_indices': [12]},
                2: {'name': 'leg', 'feature_indices': [0, 1, 5, 6, 7, 4, 10], 'qfrc_actuator_indices': [13]}
            }
            if self.is_critic:
                for key, value in self.nodes_dict.items():
                    self.nodes_dict[key]['feature_indices'].extend(value['qfrc_actuator_indices'])
                
            node_2_node = torch.tensor([[0, 1],
                                        [1, 2]])

            self.edge_index = torch.cat([
                node_2_node, node_2_node.flip(0)
            ], dim=1)

            self.joint_node_mapping = [
                 [0], [1], [2]
             ]

        elif 'ant' in task:
            """
            Ant-v5
            
            ankle_4         ankle_3
                \            /
                 \          /
                aux_4     aux_3
                    \     /
                     \   /
                      root
                     /   \
                    /     \
                  aux_2   aux_1
                 /          \
                /            \
            ankle_2          ankle_1
            
            6 nodes, 17 features
            | Node Name | Node Index  | Feature Index                | Actuator Index | cfrc_ext indices      |
            |-----------|-------------|------------------------------|----------------|-----------------------|
            | root      | 0           | *range(0, 5), *range(13, 19) |                | *range(0, 5)          |
            | aux_1     | 1           | 5, 19                        | 0              | 5                     |
            | ankle_1   | 2           | 6, 20                        | 1              | 6                     |
            | aux_2     | 3           | 7, 21                        | 2              | 7                     |
            | ankle_2   | 4           | 8, 22                        | 3              | 8                     |
            | aux_3     | 5           | 9, 23                        | 4              | 9                     |
            | ankle_3   | 6           | 10, 24                       | 5              | 10                    |
            | aux_4     | 7           | 11, 25                       | 6              | 11                    |
            | ankle_4   | 8           | 12, 26                       | 7              | 12                    |
            
            | Joint Index | Node Index | 
            |-------------|------------|
            | 0           | 0, 7       |
            | 1           | 0, 8       |
            | 2           | 0, 1       |
            | 3           | 0, 2       |
            | 4           | 0, 3       |
            | 5           | 0, 4       |
            | 6           | 0, 5       |
            | 7           | 0, 6       |
            """
            # Total number of nodes
            self.num_nodes = 9
            self.nodes_dict = {
                0: {'name': 'torso', 'feature_indices': [*range(0, 6), *range(13, 19)], 'qfrc_actuator_indices': [*range(0, 5)]},
                1: {'name': 'aux_1', 'feature_indices': [5, 19], 'qfrc_actuator_indices': [5]},
                2: {'name': 'ankle_1', 'feature_indices': [6, 20], 'qfrc_actuator_indices': [6]},
                3: {'name': 'aux_2', 'feature_indices': [7, 21], 'qfrc_actuator_indices': [7]},
                4: {'name': 'ankle_2', 'feature_indices': [8, 22], 'qfrc_actuator_indices': [8]},
                5: {'name': 'aux_3', 'feature_indices': [9, 23], 'qfrc_actuator_indices': [9]},
                6: {'name': 'ankle_3', 'feature_indices': [10, 24], 'qfrc_actuator_indices': [10]},
                7: {'name': 'aux_4', 'feature_indices': [11, 25], 'qfrc_actuator_indices': [11]},
                8: {'name': 'ankle_4', 'feature_indices': [12, 26], 'qfrc_actuator_indices': [12]},
            }
            for key, value in self.nodes_dict.items():
                # cfrc_ext
                if len(self.nodes_dict[key]['qfrc_actuator_indices']) > 0:
                    for idx in self.nodes_dict[key]['qfrc_actuator_indices']:
                        self.nodes_dict[key]['feature_indices'].extend(range(27 + idx * 6, 27 + (idx + 1) * 6))
                if self.is_critic:
                    # current joint actions
                    if int(key) != 0:    
                        self.nodes_dict[key]['feature_indices'].extend([105 + int(key) - 1])
                
            node_2_node = torch.tensor([[0, 1, 0, 3, 0, 5, 0, 7],
                                        [1, 2, 3, 4, 5, 6, 7, 8]])

            self.edge_index = torch.cat([
                node_2_node, node_2_node.flip(0)
            ], dim=1)
            
            self.num_joints = 8

            self.joint_node_mapping = [
                [0, 7], [0, 8], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]
            ]
        
        # Create separate encoders for base and joint nodes
        self.node_feature_extractor = NodeTypeSpecificMLP(self.nodes_dict, hidden_channels, activation_fn)

        # Create standard graph convolutions for each layer
        self.convs = torch.nn.ModuleList()
        # self.norms = torch.nn.ModuleList()  # Add layer norms
        # self.dropout = nn.Dropout(p=0.1)    # Add dropout

        for _ in range(num_layers):
            # Use standard GraphConv for all edges
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
            
        # for _ in range(num_layers):
        #     self.convs.append(GraphConv(hidden_channels, hidden_channels))
        #     self.norms.append(LayerNorm(hidden_channels))  # Per-node normalization
        
        if 'ant' in task or 'humanoid' in task:
            if self.is_critic:
                self.decoder = nn.Sequential(
                    Linear(hidden_channels * self.num_nodes, hidden_channels),
                    self.activation,
                    Linear(hidden_channels, 1)
                )
            else:
                self.decoder = nn.Sequential(
                    Linear(hidden_channels, int(hidden_channels/2)),
                    self.activation,
                    Linear(int(hidden_channels/2), 2)
                )
        elif 'hopper' in task:
            if self.is_critic:
                self.decoder = nn.Sequential(
                    Linear(hidden_channels * self.num_nodes, hidden_channels),
                    self.activation,
                    Linear(hidden_channels, 1)
                )
            else:
                self.decoder = nn.Sequential(
                    Linear(hidden_channels * self.num_nodes, hidden_channels),
                    self.activation,
                    Linear(hidden_channels, self.num_joints * 2)
                )

        # Create batched edge indices for common batch sizes
        self.edge_index_batch_default = self._create_edge_index_batch(self.batch_size).to(self.device)

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
    
    def _obs_to_graph_features(self, obs_all):
        """
        Convert observations into features for each node in a batch of graphs.
        """
        batch_size = obs_all.shape[0]
        
        x_feature_dict = {}
        for key, value in self.nodes_dict.items():
            x_feature_dict[key] = obs_all[:, value['feature_indices']]
            
        if batch_size == self.batch_size:
            edge_index = self.edge_index_batch_default
        elif batch_size == 1:
            edge_index = self.edge_index.to(self.device)
        else:
            edge_index = self._create_edge_index_batch(batch_size)
            
        return x_feature_dict, edge_index
    
    def _get_joint_features(self, x):
        '''
        x: [batch_size, num_nodes, hidden_channels]
        return: [batch_size * num_joints, hidden_channel]
        '''
        batch_size, _, hidden_channels = x.shape
        joint_features = []
        for mapping in self.joint_node_mapping:
            if len(mapping) == 2:
                joint_features.append((x[:, mapping[0], :] + x[:, mapping[1], :]) / 2)
            elif len(mapping) == 1:
                joint_features.append(x[:, mapping[0], :])
        joint_features = torch.cat(joint_features, dim=1)
        return joint_features.reshape(batch_size * self.num_joints, hidden_channels)

    def forward(self, obs):
        '''
        obs: 
            actor: [batch_size, 348]
            critic: [batch_size, 348 + 17]
        '''
        batch_size = obs.shape[0]

        x_feature_dict, edge_index = self._obs_to_graph_features(obs)

        # Initial feature encoding - separate for base and joint nodes
        x = self.node_feature_extractor(x_feature_dict)
        
        # Message passing layers
        for conv in self.convs:
            # Apply convolution
            x_new = conv(x, edge_index)
            x_new = self.activation(x_new)
            x = x_new
        
        # for conv, norm in zip(self.convs, self.norms):
        #     residual = x  # Save for skip connection
        #     x = conv(x, edge_index)
        #     x = norm(x)               # Normalize node features
        #     x = self.activation(x)
        #     x = self.dropout(x)
        #     x = x + residual          # Residual connection

        
        # For critic, use all node embeddings
        if 'ant' in self.task or 'humanoid' in self.task:
            if self.is_critic:
                x_reshaped = x.reshape(batch_size, self.num_nodes, -1).flatten(1)  # [batch_size, num_nodes * hidden_channels]
                final_output = self.decoder(x_reshaped)  # [batch_size, 17*2]
            else: 
                x_reshaped = x.reshape(batch_size, self.num_nodes, -1)  # [batch_size, num_nodes, hidden_channels]
                final_output = self.decoder(self._get_joint_features(x_reshaped)).reshape(batch_size, self.num_joints, 2).permute(0, 2, 1).flatten(1)  # [batch_size, 2 * 17 / 1]
        else:
            x_reshaped = x.reshape(batch_size, self.num_nodes, -1).flatten(1)  # [batch_size, num_nodes * hidden_channels]
            final_output = self.decoder(x_reshaped)  # [batch_size, 17*2]

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
