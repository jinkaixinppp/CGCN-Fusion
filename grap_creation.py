
import torch
import torch.nn.functional as F
import numpy as np
from torch import  nn

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self,batch_size,image_size1=256, patch_size1=16, in_c=1, embed_dim=64, norm_layer=None):
        super().__init__()
        self.batch_size=batch_size
        self.image_size = image_size1*image_size1
        self.patch_size = patch_size1*patch_size1
        self.grid_size = image_size1//patch_size1
        self.num_patches = self.grid_size * self.grid_size

        # The input tensor is divided into patches using 16x16 convolution
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size1, stride=patch_size1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Calculate the index of the maximum pixel in each patch.
        self.maxpool = nn.MaxPool2d(kernel_size=patch_size1, stride=patch_size1)

    def get_Q(self, img, p_size):

        b, c, h, w = img.shape
        self.w = w
        num_patches_per_side = h // p_size

        pixel_i, pixel_j = torch.meshgrid(torch.arange(h), torch.arange(w))
        patch_i = pixel_i // p_size
        patch_j = pixel_j // p_size

        pixel_idx = (patch_i * num_patches_per_side + patch_j) * p_size ** 2 + (pixel_i % p_size) * p_size + (
                pixel_j % p_size)
        pixel_idx=pixel_idx.to(device)
        Q = torch.zeros(self.batch_size, self.num_patches, h * w, dtype=torch.float,device=device) # 初始化Q矩阵
        Q = Q.view(self.batch_size, self.num_patches, -1).to(device)
        pixel_idx = pixel_idx.view(self.batch_size, -1).to(device)

        Q.scatter_(2, pixel_idx.unsqueeze(1), 1)  # Set the mapping relationship in the Q matrix.

        self.Q = Q
        return self.Q

    def compute_H(self, feature):

        B,C,H,W=feature.shape

        feature = torch.flatten(feature, start_dim=2, end_dim=3)

        # Use the torch.argmax function to find the index of the maximum pixel value in each node.
        max_index = torch.argmax(self.Q, dim=-1) .to(device) #[B,max_supiex]
        max_index=max_index.to(device)

        # Use the torch.reshape function to change the shape of the original image features to [B, H * W, C].
        feature_reshaped = feature.permute(0, 2, 1)

        # Use the torch.gather function to extract corresponding features from the original image features based on the index of the maximum pixel value.
        H = torch.gather(feature_reshaped, dim=1, index=max_index.unsqueeze(dim=-1).expand(-1, -1, C))
        H=H.to(device)
        return H

    def get_position(self):

        coords = torch.zeros((self.batch_size, self.num_patches, 2),device=device)

        # Traverse each batch of images.
        for batch_idx in range(self.batch_size):

            for patch_idx in range(self.num_patches):
                # Find the position index of the maximum value in the current patch within the Q matrix.
                pixel_idx = torch.argmax(self.Q[batch_idx, patch_idx, :])

                coords[batch_idx, patch_idx, 0] = pixel_idx // self.w
                coords[batch_idx, patch_idx, 1] = pixel_idx % self. w

        self.coords = coords
        return self.coords


    def forward(self, img):

        img = img.to(dtype=torch.float32,device=device)
        self.proj = self.proj.to(dtype=torch.float32,device=device)

        # Compute patch embeddings
        x = self.proj(img).flatten(2).transpose(1, 2)  # shape: (B, num_node, embed_dim)
        x = self.norm(x)

        return x

class Grap_creation(object):
     def __init__(self,x):

         Batch,node,_=x.shape
         self.Batch=Batch
         self.node=node
         self.data=x      #[batch,num_node,embed_dim]

     def get_A(self, patch_features, k):
         device = patch_features.device
         batch_size, num_patches, feature_dim = patch_features.size()

         # Calculate Euclidean distance
         distances = torch.cdist(patch_features, patch_features, p=2).to(device)

         # Obtain the top k+1 nearest neighbors
         topk_values, topk_indices = torch.topk(distances, k + 1, largest=False)


         A = torch.zeros(batch_size, num_patches, num_patches, device=device)

         # Retrieve the index of neighbor relationships
         row_indices = torch.arange(num_patches).unsqueeze(0).repeat(batch_size, 1).unsqueeze(2).to(device)
         col_indices = topk_indices[:, :, 1:].to(device)

         # Using indexes to set neighbor relationships
         A.scatter_(2, row_indices, 1.0)
         A.scatter_(2, col_indices, 1.0)

         return A


class Grapcreation(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.b,self.c,self.h, self.w, = self.data.shape
        self.Grap_embed = PatchEmbed(self.b,image_size1=self.h, patch_size1=16, in_c=self.c, embed_dim=64, norm_layer=None)

    def forward(self, img, feature):
        x = self.Grap_embed(img)

        Q=self.Grap_embed.get_Q(img,p_size=16)
        H= self.Grap_embed.compute_H(feature)
        position=self.Grap_embed.get_position()
        Graph =Grap_creation(x)
        A = Graph.get_A(x, k=12)


        return H, A,Q,position

