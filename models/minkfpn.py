# Author: Jacek Komorowski
# Warsaw University of Technology

import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from layers.eca_block import ECABasicBlock

from models.resnet import ResNetBase
from models.resnet import make_layer
import torch

from tools.utils import set_seed
set_seed(7)







class MinkFPN(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    # in_channels=1, out_channels=128, 1, ECABasicBlock, [1,1,1], [32,64,64]
    def __init__(self, in_channels, out_channels, num_top_down=1, conv0_kernel_size=5, block=BasicBlock,
                 layers=(1, 1, 1), planes=(32, 64, 64)):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()    # Bottom-up convolutional blocks with stride=2
        self.bns = nn.ModuleList()       # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()   # Bottom-up blocks
        self.tconvs = nn.ModuleList()   # Top-down tranposed convolutions
        self.conv1x1s = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size,
                                             dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D))
            self.bns.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1s.append(ME.MinkowskiConvolution(self.planes[-1 - i], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
            self.tconvs.append(ME.MinkowskiConvolutionTranspose(self.lateral_dim, self.lateral_dim, kernel_size=2,
                                                                stride=2, dimension=D))
        # There's one more lateral connection than top-down TConv blocks
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            self.conv1x1s.append(ME.MinkowskiConvolution(self.planes[-1 - self.num_top_down], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
        else:
            # Lateral connection from Con0 block
            self.conv1x1s.append(ME.MinkowskiConvolution(self.planes[0], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))

        self.relu = ME.MinkowskiReLU(inplace=True)

        


    def forward(self, x):
        identity = x
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger stride)
        feature_maps = []
        x = self.conv0(x) # 32
        x = self.bn0(x)  
        x = self.relu(x)
        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bns, self.blocks)):
            x = conv(x)     # 32 # Decreases spatial resolution (conv stride=2)
            x = bn(x)
            x = self.relu(x)
            x = block(x)
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down

        x = self.conv1x1s[0](x)

        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)        # Upsample using transposed convolution
            x = x + self.conv1x1s[ndx+1](feature_maps[-ndx - 1])

        return x











class GeneralMinkFPN(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    # in_channels=1, out_channels=128, 1, ECABasicBlock, [1,1,1], [32,64,64]
    def __init__(self, in_channels, out_channels, num_top_down=1, conv0_kernel_size=5, block=ECABasicBlock,
                 layers=(1, 1, 1, 1), planes=(32, 64, 64, 64)):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.out_channels = out_channels
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

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


        self.conv1x1 = ME.MinkowskiConvolution(self.planes[-1], self.out_channels, kernel_size=1, stride=1, dimension=D)



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
            

        return feature_maps



    def forward(self, x):


        feature_maps = self.forward_backbone(x)

        x = self.conv1x1(feature_maps[-1])

        # decomposed_features = x.decomposed_features
        # a = torch.nn.utils.rnn.pad_sequence(decomposed_features, batch_first=True)


        return x


