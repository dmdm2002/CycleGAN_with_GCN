import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import matplotlib as plt
import math
import numpy as np

from Modeling.GenGraph import Watts_Strogatz
from torchsummary import summary


# class GCNConv(nn.Module):
#     def __init__(self, A, in_channels, out_channels):
#         super(GCNConv, self).__init__()
#
#         # torch.eye : Matrix의 대각 성분만 1이고 나머지는 0인 Matrix를 생성한다.
#         self.A_hat = A + torch.eye(A.size(0))
#
#         # torch.diag : Matrix의 대각 성분을 가져온다.
#         self.D = torch.diag(torch.sum(A, 1))
#         self.D = self.D.inverse().sqrt()
#
#         # torch.mm : multiplication
#         # GCN -> (H * W)
#         self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
#         self.W = nn.Parameter(torch.rand(in_channels, out_channels, requires_grad=True))
#
#     def forward(self, X):
#         out = torch.relu(torch.mm(torch.mm(self.A_hat, X), self.W))
#
#         return out
#
#
# class Net(torch.nn.Module):
#     def __init__(self, A, nfeat, nhid, nout):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(A, nfeat, nhid)
#         self.relu = nn.LeakyReLU(0.2)
#         self.conv2 = GCNConv(A, nhid, nout)
#
#     def forward(self, X):
#         H = self.conv1(X)
#         H = self.relu(H)
#         H2 = self.conv2(H)
#         return H2 + X

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        #print(adj.size())
       #print(support.size())
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# def get_Adj(adj_file):
#     import scipy.io as spio
#     data = spio.loadmat(adj_file)
#     data = data['FULL'].astype(np.float32)
#     return data

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class ResGCN(nn.Module):
    def __init__(self, features, adj):
        super(ResGCN, self).__init__()
        self.adj = adj
        self.A = nn.Parameter(torch.from_numpy(self.adj).float())
        self.relu = nn.LeakyReLU(0.2)
        self.graph_conv1 = GraphConvolution(features, features)
        self.graph_conv2 = GraphConvolution(features, features)

    def forward(self, input):
        adj = gen_adj(self.A).detach()
        res_g = self.graph_conv1(input, adj)
        res_g = self.relu(res_g)
        res_g = self.graph_conv1(res_g, adj)
        return input + res_g


class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(f, f, 3, 1, 1),
            nn.InstanceNorm2d(f), nn.ReLU(),
            nn.Conv2d(f, f, 3, 1, 1),
        )
        self.norm = nn.InstanceNorm2d(f)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)+x))