









import torch.nn as nn
import MinkowskiEngine as ME
from layers.eca_block import ECABasicBlock

from models.resnet import ResNetBase


from layers.pooling import MinkGeM as MinkGeM






class GeneralMinkFPN(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    # in_channels=1, out_channels=128, 1, ECABasicBlock, [1,1,1], [32,64,64]
    def __init__(self, in_channels, out_channels, num_top_down=1, conv0_kernel_size=5, block=ECABasicBlock,
                 layers=(1, 1, 1), planes=(32, 64, 64)):
        # assert len(layers) == len(planes)
        # assert 1 <= len(layers)
        # assert 0 <= num_top_down <= len(layers)
        # self.out_channels = out_channels
        # self.num_bottom_up = len(layers)
        # self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        # self.lateral_dim = out_channels
        # self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)

    def network_initialization(self, in_channels, out_channels, D):
        # assert len(self.layers) == len(self.planes)
        # assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()    # Bottom-up convolutional blocks with stride=2
        self.bns = nn.ModuleList()       # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()   # Bottom-up blocks


        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv1 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)

        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D))
            self.bns.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))


        self.conv1x1 = ME.MinkowskiConvolution(self.planes[-1], 128, kernel_size=1, stride=1, dimension=D)

        self.mink_gem = MinkGeM(input_dim=128)


    def forward_backbone(self, x):
        feature_maps = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer_id, (conv, bn, block) in enumerate(zip(self.convs, self.bns, self.blocks)):
            x = conv(x)     # Decreases spatial resolution (conv stride=2)
            x = bn(x)
            x = self.relu(x)
            x = block(x)
            feature_maps.append(x)
            

        return x, feature_maps



    def forward(self, data_dict):
        
        x = ME.SparseTensor(features=data_dict['features'], coordinates=data_dict['coords'])


        x, feature_maps = self.forward_backbone(x)

        x = self.conv1x1(x)

        x_gem = self.mink_gem(x)


    
        return {
            'x_feat':x,
            'embedding':x_gem
        }


