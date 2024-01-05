# Author: Jacek Komorowski
# Warsaw University of Technology

# Model processing LiDAR point clouds and RGB images

import torch
import torch.nn as nn
import torchvision.models as TVmodels
from network.image_pool_fns import ImageGeM
from network.image_pool_fns import ImageCosPlace
from network.image_pool_fns import ImageNetVLAD
from network.image_pool_fns import ImageConvAP

import torch.nn.functional as F

from tools.utils import set_seed
set_seed(7)

class MinkLocMultimodal(torch.nn.Module):
    def __init__(self, cloud_fe, cloud_fe_size, image_fe, image_fe_size, 
                 fuse_method, dropout_p: float = None, final_block: str = None):
        super().__init__()



        self.cloud_fe = cloud_fe


        self.image_fe = image_fe


        self.fuse_method = fuse_method




    def forward(self, batch):
        y = {}
        if self.image_fe is not None:
            image_embedding, imagefe_output_dict = self.image_fe(batch)
            assert image_embedding.dim() == 2
            y['image_embedding'] = image_embedding
            for _k, _v in imagefe_output_dict.items():
                y[_k] = _v

        if self.cloud_fe is not None:
            cloud_embedding = self.cloud_fe(batch)['embedding']
            assert cloud_embedding.dim() == 2
            y['cloud_embedding'] = cloud_embedding


        assert cloud_embedding.shape[0] == image_embedding.shape[0]


        if self.fuse_method == 'cat':
            x = torch.cat([cloud_embedding, image_embedding], dim=1)
        elif self.fuse_method == 'add':
            assert cloud_embedding.shape == image_embedding.shape
            x = cloud_embedding + image_embedding
        else:
            raise NotImplementedError
            

        y['embedding'] = x

        
        return y











class ResnetFPN(torch.nn.Module):
    def __init__(self, out_channels: int, lateral_dim: int, layers=[64, 64, 128, 256, 512], fh_num_bottom_up: int = 5,
                 fh_num_top_down: int = 2, add_fc_block: bool = False, pool_method='gem',
                 add_basicblock: bool = True, num_basicblocks: int = 2):
        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        super().__init__()
        assert 0 < fh_num_bottom_up <= 5
        assert 0 <= fh_num_top_down < fh_num_bottom_up

        self.out_channels = out_channels
        self.lateral_dim = lateral_dim
        self.fh_num_bottom_up = fh_num_bottom_up
        self.fh_num_top_down = fh_num_top_down
        self.add_fc_block = add_fc_block
        self.layers = layers    # Number of channels in output from each ResNet block
        self.pool_method = pool_method.lower()
        self.add_basicblock = add_basicblock
        self.num_basicblocks = num_basicblocks


        model = TVmodels.resnet18(pretrained=True)



        # Last 2 blocks are AdaptiveAvgPool2d and Linear (get rid of them)
        self.resnet_fe = nn.ModuleList(list(model.children())[:3+self.fh_num_bottom_up])

        # Lateral connections and top-down pass for the feature extraction head
        self.fh_conv1x1 = nn.ModuleDict()       # 1x1 convolutions in lateral connections to the feature head
        self.fh_tconvs = nn.ModuleDict()        # Top-down transposed convolutions in feature head
        self.fh_tbasicblocks = nn.ModuleDict()  # Top-down basic blocks in feature head
        for i in range(self.fh_num_bottom_up - self.fh_num_top_down, self.fh_num_bottom_up):
            self.fh_conv1x1[str(i + 1)] = nn.Conv2d(in_channels=layers[i], out_channels=self.lateral_dim, kernel_size=1)
            self.fh_tconvs[str(i + 1)] = torch.nn.ConvTranspose2d(in_channels=self.lateral_dim,
                                                                  out_channels=self.lateral_dim,
                                                                  kernel_size=2, stride=2)



        # One more lateral connection
        temp = self.fh_num_bottom_up - self.fh_num_top_down
        self.fh_conv1x1[str(temp)] = nn.Conv2d(in_channels=layers[temp-1], out_channels=self.lateral_dim, kernel_size=1)



        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        if self.pool_method == 'gem':
            self.pool = GeM()
        elif self.pool_method == 'spoc':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pool_method == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError("Unknown pooling method: {}".format(self.pool_method))

        if self.add_fc_block:
            self.fc = torch.nn.Linear(in_features=self.lateral_dim, out_features=self.out_channels)

    def forward(self, batch):
        x = batch['images']
        feature_maps = {}

        # 0, 1, 2, 3 = first layers: Conv2d, BatchNorm, ReLu, MaxPool2d
        x = self.resnet_fe[0](x)
        x = self.resnet_fe[1](x)
        x = self.resnet_fe[2](x)
        x = self.resnet_fe[3](x)
        feature_maps["1"] = x

        # sequential blocks, build from BasicBlock or Bottleneck blocks
        for i in range(4, self.fh_num_bottom_up+3):
            x = self.resnet_fe[i](x)
            feature_maps[str(i-2)] = x

        assert len(feature_maps) == self.fh_num_bottom_up
        # x is (batch_size, 512, H=20, W=15) for 640x480 input image

        # FEATURE HEAD TOP-DOWN PASS
        xf = self.fh_conv1x1[str(self.fh_num_bottom_up)](feature_maps[str(self.fh_num_bottom_up)])
        if self.add_basicblock:
            xf = self.fh_tbasicblocks[str(self.fh_num_bottom_up)](xf)

        for i in range(self.fh_num_bottom_up, self.fh_num_bottom_up - self.fh_num_top_down, -1):
            xf = self.fh_tconvs[str(i)](xf)        # Upsample using transposed convolution
            xf = xf + self.fh_conv1x1[str(i-1)](feature_maps[str(i - 1)])

        x = self.pool(xf)
        # x is (batch_size, 512, 1, 1) tensor

        x = torch.flatten(x, 1)
        # x is (batch_size, 512) tensor

        if self.add_fc_block:
            x = self.fc(x)

        # (batch_size, feature_size)
        assert x.shape[1] == self.out_channels

        
        return x


# GeM code adapted from: https://github.com/filipradenovic/cnnimageretrieval-pytorch

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)












# ----------------------------------  ResNetFPNv2 ----------------------------------
class ResNetFPNv2(torch.nn.Module):
    def __init__(self, 
                 image_fe,
                 image_pool_method, # GeM
                 image_useallstages, # True
                 output_dim,
                 ):
        super().__init__()
        '''
        resnet [64,64,128,256,512]
        convnext [96,96,192,384,768]
        swin [96,96,192,384,768]
        swin_v2 [96,96,192,384,768]
        '''


        self.image_fe = image_fe
        self.image_pool_method = image_pool_method




        self.image_useallstages = image_useallstages

        # -- resnet
        if self.image_fe == 'resnet18':
            self.model = TVmodels.resnet18(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 512
            else:
                self.last_dim = 256
        elif self.image_fe == 'resnet34':
            self.model = TVmodels.resnet34(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 512
            else:
                self.last_dim = 256
        elif self.image_fe == 'resnet50':
            self.model = TVmodels.resnet50(weights='IMAGENET1K_V2')
            if self.image_useallstages:
                self.last_dim = 2048
            else:
                self.last_dim = 1024
        elif self.image_fe == 'resnet101':
            self.model = TVmodels.resnet101(weights='IMAGENET1K_V2')
            if self.image_useallstages:
                self.last_dim = 2048
            else:
                self.last_dim = 1024
        elif self.image_fe == 'resnet152':
            self.model = TVmodels.resnet152(weights='IMAGENET1K_V2')
            if self.image_useallstages:
                self.last_dim = 2048
            else:
                self.last_dim = 1024


        # -- convnext
        elif self.image_fe == 'convnext_tiny':
            self.model = TVmodels.convnext_tiny(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 768
            else:
                self.last_dim = 384
        elif self.image_fe == 'convnext_small':
            self.model = TVmodels.convnext_small(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 768
            else:
                self.last_dim = 384


        # -- swin
        elif self.image_fe == 'swin_t':
            self.model = TVmodels.swin_t(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 768
            else:
                self.last_dim = 384
        elif self.image_fe == 'swin_s':
            self.model = TVmodels.swin_s(weights='IMAGENET1K_V1')
            self.last_dim = 384
        elif self.image_fe == 'swin_v2_t':
            self.model = TVmodels.swin_v2_t(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 768
            else:
                self.last_dim = 384
        elif self.image_fe == 'swin_v2_s':
            self.model = TVmodels.swin_v2_s(weights='IMAGENET1K_V1')
            self.last_dim = 384





        self.conv1x1 = nn.Conv2d(self.last_dim, output_dim, kernel_size=1)


        self.image_gem = ImageGeM() # *1
        self.imagecosplace =  ImageCosPlace(output_dim, output_dim) # *1
        self.imageconvap = ImageConvAP(output_dim, output_dim) # *4
        self.imagenetvlad = ImageNetVLAD(clusters_num=64,
                                         dim=output_dim) # *4

        

    def forward_resnet(self, x):
        fe_output_dict = {}
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x_avgpool = F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze(3).squeeze(2)
        fe_output_dict['image_layer1'] = x
        fe_output_dict['image_layer1_avgpool'] = x_avgpool

        x = self.model.layer2(x)
        x_avgpool = F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze(3).squeeze(2)
        fe_output_dict['image_layer2'] = x
        fe_output_dict['image_layer2_avgpool'] = x_avgpool

        x = self.model.layer3(x)
        x_avgpool = F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze(3).squeeze(2)
        fe_output_dict['image_layer3'] = x
        fe_output_dict['image_layer3_avgpool'] = x_avgpool

        if self.image_useallstages:
            x = self.model.layer4(x)
            x_avgpool = F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze(3).squeeze(2)
            fe_output_dict['image_layer4'] = x
            fe_output_dict['image_layer4_avgpool'] = x_avgpool

        return x, fe_output_dict



    def forward_convnext(self, x):
        layers_list = list(self.model.features.children())
        assert len(layers_list)==8
        if not self.image_useallstages:
            layers_list = layers_list[:-2]
        else:
            layers_list = layers_list

        for i in range(len(layers_list)):
            layer = layers_list[i]
            x = layer(x)
        return x

    
    def forward_swin(self, x):
        layers_list = list(self.model.features.children())
        if not self.image_useallstages:
            layers_list = layers_list[:-2]
        else:
            layers_list = layers_list
        for i in range(len(layers_list)):
            layer = layers_list[i]
            x = layer(x)
        x = x.permute(0,3,1,2)
        return x






    def forward(self, data_dict):


        x = data_dict['images']
        fe_output_dict = {}

        
        if self.image_fe in ['resnet18','resnet34','resnet50','resnet101','resnet152']:
            x, fe_output_dict = self.forward_resnet(x)
        elif self.image_fe in ['convnext_tiny','convnext_small']:
            x = self.forward_convnext(x)
        elif self.image_fe in ['swin_t','swin_s']:
            x = self.forward_swin(x)
        elif self.image_fe in ['swin_v2_t','swin_v2_s']:
            x = self.forward_swin(x)
        else:
            raise NotImplementedError
        

        x_feat_256 = x
        x_feat_256 = self.conv1x1(x_feat_256)


        if self.image_pool_method == 'GeM':
            embedding = self.image_gem(x_feat_256)

        elif self.image_pool_method == 'ConvAP':
            embedding = self.imageconvap(x_feat_256)

        elif self.image_pool_method == 'CosPlace':
            embedding = self.imagecosplace(x_feat_256)

        elif self.image_pool_method == 'NetVLAD':
            embedding = self.imagenetvlad(x_feat_256)
        
        else:
            raise NotImplementedError
        


        return embedding, fe_output_dict
    

    




