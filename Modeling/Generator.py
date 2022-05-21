import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from Modeling.GenGraph import Watts_Strogatz
# from Modeling.Layers import GCNConv, Net, ResBlock
from Modeling.Layers import GraphConvolution, ResGCN, ResBlock
from torchsummary import summary


class Gen(nn.Module):
    def __init__(self, blocks=6):
        super(Gen, self).__init__()

        # init = nn.R
        encoder_1 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64)
        ]
        encoder_2 = [
            self.__conv_block(64, 128)
        ]
        encoder_3 = [
            self.__conv_block(128, 256)
        ]

        self.downsample_graph = nn.Conv2d(256, 256, kernel_size=3, stride=10, padding=1)

        res_blocks = [
            ResBlock(256) for _ in range(blocks)
        ]

        _adj = Watts_Strogatz(256, 128, 0.5)()

        self.A = nn.Parameter(torch.from_numpy(_adj).float())
        GCN_blocks = [
            ResGCN(256, _adj) for _ in range(blocks)
        ]
        self.graph_convhead = GraphConvolution(1, 256)
        self.graph_convtail = GraphConvolution(256, 1)

        self.upsample_graph = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=10, padding=1,
                                                 output_padding=5)
        decoder_1 = [
            self.__conv_block(256, 128, upsample=True),
        ]

        decoder_2 = [
            self.__conv_block(128, 64, upsample=True)
        ]

        decoder_3 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh(),
        ]

        self.relu = nn.LeakyReLU(0.2)
        self.encoder_1 = nn.Sequential(*encoder_1)
        self.encoder_2 = nn.Sequential(*encoder_2)
        self.encoder_3 = nn.Sequential(*encoder_3)

        self.res_blocks = nn.Sequential(*res_blocks)
        self.GCN_blocks = nn.Sequential(*GCN_blocks)

        self.decoder_1 = nn.Sequential(*decoder_1)
        self.decoder_2 = nn.Sequential(*decoder_2)
        self.decoder_3 = nn.Sequential(*decoder_3)

    def __conv_block(self, in_features, out_features, upsample=False):
        if upsample:
            conv = nn.ConvTranspose2d(in_features, out_features, 3, 2, 1, output_padding=1)

        else:
            conv = nn.Conv2d(in_features, out_features, 3, 2, 1)

        return nn.Sequential(
            conv,
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)

        return adj

    def forward(self, input):
        x = self.encoder_1(input)
        x = self.encoder_2(x)
        x = self.encoder_3(x)

        x = self.res_blocks(x)

        yin = self.downsample_graph(x)
        yin = yin.permute(0,2,3,1)
        yin = yin.unsqueeze(4)

        adj = self.gen_adj(self.A).detach()
        res_g = self.graph_convhead(yin, adj)  # make(yin, adj) as dict to let it available in nn.Sequential
        res_g = self.GCN_blocks(res_g)
        res_g = self.graph_convtail(res_g, adj)
        res_g = self.relu(res_g)

        res_g = res_g.squeeze(4)
        yout_g = res_g.permute(0, 3, 1, 2)
        yout_g = self.upsample_graph(yout_g)

        x = yout_g + x

        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)

        return x
