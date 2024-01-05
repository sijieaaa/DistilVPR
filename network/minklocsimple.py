# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from models.minkfpn import MinkFPN
import layers.pooling as pooling
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from layers.eca_block import ECABasicBlock

import numpy as np


from tools.options import Options
args = Options().parse()
from tools.utils import set_seed
set_seed(7)


class MinkLocSimple(torch.nn.Module):
    def __init__(self, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size,
                 block='BasicBlock', pooling_method='GeM', linear_block=False, dropout_p=None):
        # block: Type of the network building block: BasicBlock or SEBasicBlock
        # add_linear_layers: Add linear layers at the end
        # dropout_p: dropout probability (None = no dropout)

        super().__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size    # Size of local features produced by local feature extraction block
        self.output_dim = output_dim        # Dimensionality of the global descriptor produced by pooling layer
        self.block = block

        if block == 'BasicBlock':
            block_module = BasicBlock
        elif block == 'Bottleneck':
            block_module = Bottleneck
        elif block == 'ECABasicBlock':
            block_module = ECABasicBlock
        else:
            raise NotImplementedError('Unsupported network block: {}'.format(block))

        self.pooling_method = pooling_method
        self.linear_block = linear_block
        self.dropout_p = dropout_p
        self.backbone = MinkFPN(in_channels=in_channels, out_channels=self.feature_size, num_top_down=num_top_down,
                                conv0_kernel_size=conv0_kernel_size, block=block_module, layers=layers, planes=planes)

        self.pooling = pooling.PoolingWrapper(pool_method=pooling_method, in_dim=self.feature_size,
                                              output_dim=output_dim)
        self.pooled_feature_size = self.pooling.output_dim      # Number of channels returned by pooling layer

        if self.dropout_p is not None:
            self.dropout = nn.Dropout(p=self.dropout_p)
        else:
            self.dropout = None

        if self.linear_block:
            # At least output_dim neurons in intermediary layer
            int_channels = self.output_dim
            self.linear = nn.Sequential(nn.Linear(self.pooled_feature_size, int_channels, bias=False),
                                        nn.BatchNorm1d(int_channels, affine=True),
                                        nn.ReLU(inplace=True), nn.Linear(int_channels, output_dim))
        else:
            self.linear = None


    
    def forward(self, batch_dict):


        # # -- new 
        # in_field = batch_dict['in_field']
        # x = in_field.sparse()
        # x = self.backbone(x)

        

        # -- old
        x = ME.SparseTensor(features=batch_dict['features'], coordinates=batch_dict['coords'])
        x = self.backbone(x)






        return x

