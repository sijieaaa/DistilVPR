# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from models.minkfpn import MinkFPN
from models.minkfpn import GeneralMinkFPN

import layers.pooling as pooling
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from layers.eca_block import ECABasicBlock

from layers.pooling import MinkGeM as MinkGeM

# from tools.options import Options
# args = Options().parse()

from tools.utils import set_seed
set_seed(7)







class MinkLocOnly(torch.nn.Module):
    def __init__(self, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size,
                 block='BasicBlock', pooling_method='GeM', linear_block=False, dropout_p=None,
                 minkfpn='minkfpn'):
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


        self.backbone = MinkFPN(in_channels=in_channels, out_channels=feature_size, num_top_down=num_top_down,
                                conv0_kernel_size=conv0_kernel_size, block=block_module, layers=layers, planes=planes)

        # self.conv1x1 = nn.Linear(feature_size, output_dim)


        self.pooling = pooling.PoolingWrapper(pool_method=pooling_method, 
                                              in_dim=feature_size,
                                              output_dim=feature_size)

        # self.conv1x1 = nn.Linear(output_dim, output_dim)


    def forward(self, batch):
        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        x = ME.SparseTensor(features=batch['features'], coordinates=batch['coords'])
        x = self.backbone(x)

        



        x_feat = x
        # x = self.conv1x1(x.F)
        # x = ME.SparseTensor(features=x, 
        #                     # coordinates=x_feat.C, 
        #                     coordinate_manager=x_feat.coordinate_manager, 
        #                     coordinate_map_key=x_feat.coordinate_map_key,
        #                     device=x_feat.device)
        x_gem = self.pooling(x)
        # x_gem = self.conv1x1(x_gem)

        
        return {
            'output_cloud_feat': x_feat,
            'output_cloud_gem': x_gem,

            'embedding':x_gem
            }






