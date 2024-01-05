# Author: Jacek Komorowski
# Warsaw University of Technology

# Model processing LiDAR point clouds and RGB images

import torch
import torch.nn as nn
import torchvision.models as TVmodels
# from TV_offline_models.swin_transformer import swin_v2_t,swin_v2_s
import MinkowskiEngine as ME

from models.minkloc import MinkLoc
from network.resnetfpn_simple import ImageGeM
from network.resnetfpn_simple import ImageConvAP
from network.resnetfpn_simple import ImageMixVPR
from network.resnetfpn_simple import ImageCosPlace
from network.resnetfpn_simple import ImageNetVLAD

from tools.utils import set_seed
set_seed(7)
# from tools.options import Options
# args = Options().parse()





class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)








class DistilImageFE(torch.nn.Module):
    def __init__(self, 
                 image_fe,
                 num_other_stage_blocks,
                 num_stage3_blocks,
                 input_type,
                 pool_method, # GeM
                 useallstages, # True
                 dataset, # oxford
                 out_channels: int=None, 
                 lateral_dim: int=None, 
                 layers=[64, 64, 128, 256, 512], 
                 fh_num_bottom_up: int = 5,
                 fh_num_top_down: int = 2, 
                 add_fc_block: bool = False, 

                 ):
        super().__init__()
        '''
        resnet [64,64,128,256,512]
        convnext [96,96,192,384,768]
        swin [96,96,192,384,768]
        swin_v2 [96,96,192,384,768]
        '''

        assert input_type in ['image','sph_cloud']
        assert dataset in ['oxford','oxfordadafusion']

        self.image_fe = image_fe
        self.num_other_stage_blocks = num_other_stage_blocks
        self.num_stage3_blocks = num_stage3_blocks
        self.input_type = input_type


        self.out_channels = out_channels
        self.pool_method = pool_method


        self.useallstages = useallstages

        # -- resnet
        if self.image_fe == 'resnet18':
            self.model = TVmodels.resnet18(weights='IMAGENET1K_V1')
            if self.useallstages:
                self.last_dim = 512
            else:
                self.last_dim = 256
        elif self.image_fe == 'resnet34':
            self.model = TVmodels.resnet34(weights='IMAGENET1K_V1')
            self.last_dim = 256
        elif self.image_fe == 'resnet50':
            self.model = TVmodels.resnet50(weights='IMAGENET1K_V2')
            self.last_dim = 1024

        # -- convnext
        elif self.image_fe == 'convnext_tiny':
            self.model = TVmodels.convnext_tiny(weights='IMAGENET1K_V1')
            if self.useallstages:
                self.last_dim = 768
            else:
                self.last_dim = 384
        elif self.image_fe == 'convnext_small':
            self.model = TVmodels.convnext_small(weights='IMAGENET1K_V1')
            self.last_dim = 384

        # -- swin
        elif self.image_fe == 'swin_t':
            self.model = TVmodels.swin_t(weights='IMAGENET1K_V1')
            self.last_dim = 384
        elif self.image_fe == 'swin_s':
            self.model = TVmodels.swin_s(weights='IMAGENET1K_V1')
            self.last_dim = 384
        elif self.image_fe == 'swin_v2_t':
            self.model = TVmodels.swin_v2_t(weights='IMAGENET1K_V1')
            self.last_dim = 384
        elif self.image_fe == 'swin_v2_s':
            self.model = TVmodels.swin_v2_s(weights='IMAGENET1K_V1')
            self.last_dim = 384



        # self.conv1x1 = nn.Conv2d(self.last_dim, 128, kernel_size=1)




        pool_dim = self.last_dim

        self.image_gem = ImageGeM() # *1
        self.imagecosplace =  ImageCosPlace(pool_dim, pool_dim) # *1
        self.imageconvap = ImageConvAP(pool_dim, pool_dim) # *4
        self.dataset = dataset
        self.image_fe = image_fe



        if self.dataset in ['oxford','oxfordadafusion']:
            if self.image_fe in ['resnet18']:
                if self.useallstages:
                    mixvpr_h, mixvpr_w = (8,10)
                else:
                    mixvpr_h, mixvpr_w = (15,20)
            elif self.image_fe in ['convnext_tiny']:
                if self.useallstages:
                    mixvpr_h, mixvpr_w = (7,10)
                else:
                    mixvpr_h, mixvpr_w = (15,20)

        elif self.dataset in ['boreas']:
            if self.image_fe in ['resnet18']:
                if self.useallstages:
                    mixvpr_h, mixvpr_w = (8,10)
                else:
                    mixvpr_h, mixvpr_w = (16,20)
            elif self.image_fe in ['convnext_tiny']:
                if self.useallstages:
                    mixvpr_h, mixvpr_w = (8,9)
                else:
                    mixvpr_h, mixvpr_w = (16,19)



        self.imagemixvpr = ImageMixVPR( # *4
            in_channels=pool_dim,
            in_h=mixvpr_h,
            in_w=mixvpr_w,
            out_channels=pool_dim,
            mix_depth=4,
            mlp_ratio=1,
            out_rows=4) # [h=16,w=20] for boreas
        self.imagenetvlad = ImageNetVLAD(clusters_num=64,
                                         dim=pool_dim) # *4

        




    def forward_resnet(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        if self.useallstages:
            x = self.model.layer4(x)

        return x



    def forward_convnext(self, x):
        layers_list = list(self.model.features.children())
        assert len(layers_list)==8
        if not self.useallstages:
            layers_list[1] = layers_list[1][:self.num_other_stage_blocks]
            layers_list[3] = layers_list[3][:self.num_other_stage_blocks]
            layers_list[5] = layers_list[5][:self.num_stage3_blocks]
            layers_list = layers_list[:-2]
        else:
            layers_list = layers_list

        for i in range(len(layers_list)):
            layer = layers_list[i]
            x = layer(x)
        return x

    
    def forward_swin(self, x):
        layers_list = list(self.model.features.children())
        if not self.useallstages:
            layers_list = layers_list[:-2]
        else:
            layers_list = layers_list
        for i in range(len(layers_list)):
            layer = layers_list[i]
            x = layer(x)
        x = x.permute(0,3,1,2)
        return x






    def forward(self, data_dict):
        if self.input_type == 'image':
            x = data_dict['images']
        elif self.input_type == 'sph_cloud':
            x = data_dict['sph_cloud']


        
        if self.image_fe in ['resnet18','resnet34','resnet50']:
            x = self.forward_resnet(x)
        elif self.image_fe in ['convnext_tiny','convnext_small']:
            x = self.forward_convnext(x)
        elif self.image_fe in ['swin_t','swin_s']:
            x = self.forward_swin(x)
        elif self.image_fe in ['swin_v2_t','swin_v2_s']:
            x = self.forward_swin(x)
        elif self.image_fe in ['efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_v2_s']:
            x = self.forward_efficientnet(x)
        elif self.image_fe in ['regnet_x_3_2gf','regnet_y_1_6gf','regnet_y_3_2gf']:
            x = self.forward_regnet(x)
        else:
            raise NotImplementedError
        

        x_feat_256 = x




        if self.pool_method == 'GeM':
            x_gem_256 = self.image_gem(x_feat_256)

        elif self.pool_method == 'ConvAP':
            x_gem_256 = self.imageconvap(x_feat_256)

        elif self.pool_method == 'MixVPR':
            x_gem_256 = self.imagemixvpr(x_feat_256)

        elif self.pool_method == 'CosPlace':
            x_gem_256 = self.imagecosplace(x_feat_256)

        elif self.pool_method == 'NetVLAD':
            x_gem_256 = self.imagenetvlad(x_feat_256)
        
        else:
            raise NotImplementedError
        


        output_dict = {
            'embedding': x_gem_256,
            'x_feat': x_feat_256,
        }


        return output_dict
    

    




