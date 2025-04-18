import numpy as np

import torch
import torch.nn as nn

from torch_geometric.nn import Linear, GraphConv
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
        
        if 'humanoid' in task:
            '''
            348 features
            | Node Name       | Node Index  | *qpos & *qvel Feature Index          | qfrc_actuator Index |
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
            
            | Joint Index | Node Index | 
            |-------------|------------|
            | 0           | 1, 2       |
            | 1           | 1, 2       |
            | 2           | 1, 2       |
            | 3           | 2, 3       |
            | 4           | 2, 3       |
            | 5           | 2, 3       |
            | 6           | 3, 4       |
            | 7           | 2, 6       |
            | 8           | 2, 6       |
            | 9           | 2, 6       |
            | 10          | 6, 7       |
            | 11          | 0, 9       |
            | 12          | 0, 9       |
            | 13          | 9, 10      |
            | 14          | 0, 11      |
            | 15          | 0, 11      |
            | 16          | 11, 12     |

            '''
            # Total number of nodes (combining base and joint)
            self.num_nodes = 13
            self.num_joints = 17
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
            
            self.joint_node_mapping = [
                [1, 2], [1, 2], [1, 2], [2, 3], [2, 3], [2, 3], [3, 4], [2, 6], [2, 6], [2, 6], [6, 7], [0, 9], [0, 9], [9, 10], [0, 11], [0, 11], [11, 12]
            ]

        elif 'halfcheetah' in task or 'walker2d' in task:
            '''
            HalfCheetah-v5
                                           front_tip
                                          /
                                         /
                                        /
                      0 bthigh ---- 3 fthigh
                      /               \
                     /                 \
                  1 bshin               4 fshin
                 /                        \
                /                          \
            2 bfoot                      5 ffoot
            
            6 nodes, 17 features
            | Node Name       | Node Index  | Feature Index          | Actuator Index |
            |-----------------|-------------|------------------------|----------------|
            | bthigh          | 0           | 0, 1, 8, 9, 10, 2, 11  | 17             |
            | bshin           | 1           | 0, 1, 8, 9, 10, 3, 12  | 18             |
            | bfoot           | 2           | 0, 1, 8, 9, 10, 4, 13  | 19             |
            | fthigh          | 3           | 0, 1, 8, 9, 10, 5, 14  | 20             |
            | fshin           | 4           | 0, 1, 8, 9, 10, 6, 15  | 21             |
            | ffoot           | 5           | 0, 1, 8, 9, 10, 7, 16  | 22             |
            
            | Joint Index | Node Index | 
            |-------------|------------|
            | 0           | 0          |
            | 1           | 1          |
            | 2           | 2          |
            | 3           | 3          |
            | 4           | 4          |
            | 5           | 5          |
            '''
            self.num_nodes = 6
            self.num_joints = 6
            self.nodes_dict = {
                0: {'name': 'bthigh', 'feature_indices': [0, 1, 8, 9, 10, 2, 11], 'qfrc_actuator_indices': [17]},
                1: {'name': 'bshin', 'feature_indices': [0, 1, 8, 9, 10, 3, 12], 'qfrc_actuator_indices': [18]},
                2: {'name': 'bfoot', 'feature_indices': [0, 1, 8, 9, 10, 4, 13], 'qfrc_actuator_indices': [19]},
                3: {'name': 'fthigh', 'feature_indices': [0, 1, 8, 9, 10, 5, 14], 'qfrc_actuator_indices': [20]},
                4: {'name': 'fshin', 'feature_indices': [0, 1, 8, 9, 10, 6, 15], 'qfrc_actuator_indices': [21]},
                5: {'name': 'ffoot', 'feature_indices': [0, 1, 8, 9, 10, 7, 16], 'qfrc_actuator_indices': [22]},
            }
            if self.is_critic:
                for key, value in self.nodes_dict.items():
                    self.nodes_dict[key]['feature_indices'].extend(value['qfrc_actuator_indices'])
                
            node_2_node = torch.tensor([[0, 1, 2, 3, 4, 2, 5],
                                        [1, 2, 3, 4, 5, 3, 4]])

            self.edge_index = torch.cat([
                node_2_node, node_2_node.flip(0)
            ], dim=1)
            
            self.joint_node_mapping = [
                [0], [1], [2], [3], [4], [5]
            ]
            
        else:
            # TODO: add other robots
            print('Not implemented')
            pass
        
        # Create separate encoders for base and joint nodes
        self.node_feature_extractor = NodeTypeSpecificMLP(self.nodes_dict, hidden_channels, activation_fn)

        # Create standard graph convolutions for each layer
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # Use standard GraphConv for all edges
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
        
        if self.is_critic:
            self.decoder = nn.Sequential(
                Linear(hidden_channels * self.num_nodes, hidden_channels),
                self.activation,
                Linear(hidden_channels, 1)
            )
        else:
            # self.decoder = nn.Sequential(
            #     Linear(hidden_channels * self.num_nodes, hidden_channels),
            #     self.activation,
            #     Linear(hidden_channels, self.num_joints*2)
            # )
            self.decoder = nn.Sequential(
                Linear(hidden_channels, int(hidden_channels / 2)),
                self.activation,
                Linear(int(hidden_channels / 2), 2)
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
        
        # For critic, use all node embeddings
        if self.is_critic:
            x_reshaped = x.reshape(batch_size, self.num_nodes, -1).flatten(1)
            final_output = self.decoder(x_reshaped)  # [batch_size, 1]
        else:
            x_reshaped = x.reshape(batch_size, self.num_nodes, -1)  #.flatten(1)  # [batch_size, num_nodes, hidden_channels]
            final_output = self.decoder(self._get_joint_features(x_reshaped)).reshape(batch_size, self.num_joints, 2).permute(0, 2, 1).flatten(1)  # [batch_size, 2 * 17 / 1]

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
