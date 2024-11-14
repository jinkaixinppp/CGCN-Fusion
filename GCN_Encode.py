
import torch
import torch.nn.functional as F
import numpy as np
from torch import  nn

from bias import GraphAttnBias
from GrapConv import GraphConvolution
from grap_creation import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define DropPath class
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x / keep_prob * random_tensor
        else:
            output = x
        return output

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop_path=0.0):
        super().__init__()

        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=(1, 1)),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_features, out_channels=in_features, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_features)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        b, n, c = x.size()
        x1 = x.view(-1, c,1,1)
        x1 = self.fc1(x1)
        x2 = self.fc2(x1)
        x2 = x2.view(-1, n, c, 1)
        x = x2.reshape(b, n, c)+x
        return x

class GCNConv(nn.Module):
    #Graph convolution, used to gather information for one hop
    def __init__(self,in_channel,out_channel):
        super(GCNConv, self).__init__()
        self.conv1 = GraphConvolution(in_channel,out_channel)
        self.act = nn.LeakyReLU()
        self.drop_prob = 0.3
        self.drop = nn.Dropout(self.drop_prob)

    def forward(self, x, adj):
        x = self.drop(self.act(self.conv1(x, adj))+ x)
        return x


class GraphConvolution_Block(nn.Module):
    def __init__(self, in_channel, out_channel, hiddenFFN_channel, dropout=0.5):
        super(GraphConvolution_Block, self).__init__()
        self.gcnconv1 = GCNConv(in_channel, out_channel)
        self.ffn1 = FFN(out_channel, hiddenFFN_channel)
        self.gcnconv2 = GCNConv(out_channel, out_channel)
        self.ffn2 = FFN(out_channel, hiddenFFN_channel)
        self.gcnconv3 = GCNConv(out_channel, out_channel)
        self.ffn3 = FFN(out_channel, hiddenFFN_channel)
        self.bn = nn.BatchNorm1d(out_channel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # Block 1
        h1 = F.relu(self.gcnconv1(x, adj))  # GCNConv
        h1 = self.dropout(h1)
        h1 = F.relu(self.ffn1(h1))  # FFN
        h1 = self.dropout(h1)
        x1 = h1

        # Block 2 (Unlike Block 1, the input for Block 2 is x1 instead of x)
        h1 = F.relu(self.gcnconv2(x1, adj))
        h1 = self.dropout(h1)
        h1 = F.relu(self.ffn2(h1))
        h1 = self.dropout(h1)
        x2 = h1 + x1

        # Block 3 (Unlike Block 1, the input for Block 3 is x2 instead of x)
        h1 = F.relu(self.gcnconv3(x2, adj))  #  GCNConv
        h1 = self.dropout(h1)
        h1 = F.relu(self.ffn3(h1))  #  FFN
        h1 = self.dropout(h1)
        h = h1 + x2 + x1


        h = self.bn(h.transpose(1, 2)).transpose(1, 2)

        return h



class Self_Attention(nn.Module):
    """
    Self attention module, using a single channel GCN to estimate attention scores
    """
    def __init__(self,in_channel,non_linearity=F.softmax, temperature=1):
        super(Self_Attention,self).__init__()
        self.in_channel = in_channel
        self.non_linearity = non_linearity
        self.temperature=temperature
        self.gcn = GCNConv(in_channel,1)

    def forward(self, x, adj):
         #x = x.unsqueeze(-1) if x.dim() == 1 else x
        b,n,c=x.size()
        score = self.gcn(x,adj).squeeze()
        epsilon = 1e-8

        # Before performing normalization, replace the zero value in the denominator with the minimum value
        score += (score == 0).float() * epsilon

        score = F.normalize(score, p=2, dim=1)  # 归一化

        score *= self.temperature


        M=self.non_linearity(score, dim=1)
        M = M.view(b, n,c)

        x = x *M +x

        return x



class MultiHeadAttention(nn.Module):
        def __init__(self, num_heads, in_channel, dropout=0.1,  bias=True,):
            super(MultiHeadAttention, self).__init__()
            self.num_heads = num_heads
            self.input_dim =in_channel
            self.d_k = in_channel // num_heads

            self.q_linear = nn.Linear(in_channel, in_channel)
            self.k_linear = nn.Linear(in_channel, in_channel)
            self.v_linear = nn.Linear(in_channel,in_channel)

            self.weight_vector = nn.Parameter(torch.randn(num_heads, self.d_k))
            self.norm = nn.LayerNorm(in_channel)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x,bias, mask=None):
            # x(batch_size, num_node, channel)
            batch_size, num_node, channel=x.size()
            batch_size, n_node, _ = x.size()
            self.attn_bias=bias

            self.out_linear = nn.Linear(num_node*self.num_heads, self.d_k)

            Q = self.q_linear(x)
            K = self.k_linear(x)
            V = self.v_linear(x)

            # x 的 shape 为 (batch_size, num_node, channel)转为 (batch_size, n_heads,num_node, channel)
            Q = Q.view(batch_size, n_node, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size,n_node, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, n_node, self.num_heads, self.d_k).transpose(1, 2)

            # Calculate the similarity matrix and add bias here
            scores = torch.matmul(Q, K.transpose(-2, -1)) \
                     + self.attn_bias
            scores = scores / (self.d_k ** 0.5)

            if mask is not None:
                mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
                scores.masked_fill_(mask == 0, -1e9)

            #Calculate the attention weight matrix and add learnable weight vectors for weighting
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)


            attn_output = torch.matmul(weights, V)
            output = attn_output.view(batch_size,num_node, self.num_heads, self.d_k)


            #
            output = output.view(batch_size, self.num_heads, num_node, self.d_k)
            output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, num_node, -1)

            # [batch_size, num_nodes, num_nodes]
            output = torch.matmul(output, output.transpose(1, 2))

            return output



class GNNEncoder(nn.Module):
    #Advanced GCN encoder based on attention.
    def __init__(self):
        super(GNNEncoder, self).__init__()

        self.GrapConv1=GraphConvolution_Block(in_channel=192,out_channel=192,hiddenFFN_channel=256,dropout=0.5)
        self.self_Attn = Self_Attention(in_channel=192, non_linearity=torch.softmax)
        self.bias_proj=GraphAttnBias(num_heads=6,num_node=256,num_spatial=2,r=0.2)
        self.Mulihead=MultiHeadAttention(num_heads=6, in_channel=192, dropout=0.1,  bias=True,)
        # self.GrapConv2 =GraphConvolution_Block(in_channel=192, out_channel=192, hiddenFFN_channel=256)


    def forward(self, img,features):
        b,c,h,w=img.shape
        grapcreation=Grapcreation(img)
        H,A,Q,position=grapcreation(img,features)
        H=H.to(device)
        A = A.to(device)
        position=position.to(device)
        Q = Q.to(device)
        output=self.GrapConv1(H,A)

        output=self.self_Attn(output,A)

        # Create a zero tensor with the same shape as the H_tensor
        mask = torch.zeros_like(output,device=device)
        output_sum = torch.sum(output, dim=2)  # Sum up on the dimension of channel-dim
        idx = torch.where(output_sum == 0)[0]
        mask[idx] = 1
        padding_mask = (mask.sum(dim=2) == output.shape[2]).float().bool()  # Sum up on the dimension of channel-dim

        graph_attn_bias = self.bias_proj(position,H,A,sigma=1).contiguous()
        graph_attn_bias=graph_attn_bias.unsqueeze(0)

        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        new_A=self.Mulihead(output,graph_attn_bias).to(device)

        output=self.GrapConv1(output,new_A).to(device)

        B,node,c=output.shape
        out=output.reshape(B,c,node)

        GCNout = torch.bmm(out, Q)

        GCNout = GCNout.view(B, c,h,w) #  [B, C, H, W]


        return GCNout
"""
def main():
    path = "CT-MRI/CT/2004.png"

   
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)) 
    img_tensor = img_tensor.unsqueeze(0)  

    img = np.expand_dims(img, axis=0)  
    img = img.transpose((0, 1, 2, 3))
    print(type(img))
    print(img.shape)   #[B,H,W,C]
    model = GNNEncoder()  
    GCNout = model(img)
    print("GCNout",GCNout.shape)

 if __name__ == '__main__':
      main()

"""