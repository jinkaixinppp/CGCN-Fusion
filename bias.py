import torch
import math
import torch.nn as nn
import numpy as np
import cv2
from grap_creation import *
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphAttnBias(nn.Module):


    def __init__(self,num_heads,num_node ,num_spatial, r):

        super(GraphAttnBias, self).__init__()
        self.num_node=num_node
        self.num_heads = num_heads
        self.num_spatial=num_spatial
        # Define position encoder
        self.spatial_pos_encoder = nn.Embedding(num_node,num_heads, padding_idx=0)
        # Hyperparameters for calculating distance
        self.r=r
        #This code creates an embedding layer to embed the positional
        # relationships between virtual nodes, adding more positional
        # to improve the model's understanding and performance of graphical information
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        # Define attention bias term
        self.attn_bias = None

    def forward(self, spatial_pos, H,A,sigma):


        batch,num_node, num_spatial = spatial_pos.shape

        self.spatial_pos = spatial_pos.long()
        self.H = H


        # Calculate attention bias matrix
        if self.spatial_pos is not None:
            # Obtain the bias matrix of the spatial position vector
            spatial_pos_bias = self.spatial_pos_encoder(
                self.spatial_pos)  # size: ( batch,num_nodes, self.num_heads * self.hidden_dim)
            # Make shape changes for subsequent calculations
            spatial_pos_bias = spatial_pos_bias.view(batch, self.num_heads,num_node,
                                                     self.num_spatial)  # size: ( batch,num_heads,num_nodes,  hidden_dim)

            spatial_pos_bias_transpose = spatial_pos_bias.permute( 0,1, 3,
                                                                  2)  # size: (batch,num_heads, hidden_dim, num_nodes)

            spatial_pos_bias = torch.matmul(spatial_pos_bias,spatial_pos_bias_transpose)

        graph_attn_bias = torch.zeros(batch,self.num_heads, num_node+1, num_node+1,device=device)

        graph_attn_bias[ :,:, 1:, 1:] += spatial_pos_bias
        # reset spatial pos here

        t = self.graph_token_virtual_distance.weight.view(self.num_heads, 1)
        graph_attn_bias[:,:, 1:, 0] = graph_attn_bias[:,:, 1:, 0] + t
        graph_attn_bias[:,:, 0, :] = graph_attn_bias[:,:, 0, :] + t

        degree = torch.sum(A, dim=-1, keepdim=True)
        degree_inv_sqrt = torch.pow(degree, -0.5)

        nor_A = degree_inv_sqrt * A

        # Calculate the weighted average of neighbor node features
        neighbor_features = torch.matmul(nor_A, H)

        # Calculate the square of Euclidean distance
        euclidean_distance_sq = torch.sum((H.unsqueeze(2) - neighbor_features.unsqueeze(1)) ** 2, dim=-1)

        # Calculate Gaussian distance
        gaussian_dist = torch.exp (-euclidean_distance_sq / (2 * sigma ** 2))

        # Calculate the similarity between pixel vectors and assign it to bais
        edg_bias = gaussian_dist

        # Add a dimension representing the head after the similarity matrix
        edg_bias = edg_bias.unsqueeze(1)  # 形状为[B, head, node, node]
        edg_bias=edg_bias.expand(-1,self.num_heads,-1,-1)

        graph_attn_bias[:,:, 1:, 1:] = graph_attn_bias[:,:, 1:, 1:] + edg_bias
        ###[batch_size,num_heads,num_node,num_node]
        graph_attn_bias = graph_attn_bias[:,:, 1:num_node + 1, 1:num_node + 1]

        return graph_attn_bias




if __name__ == '__main__':
    main()
