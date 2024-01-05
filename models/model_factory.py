# Author: Jacek Komorowski
# Warsaw University of Technology

from models.minkloc import MinkLoc
from models.minkloc_multimodal import MinkLocMultimodal, ResnetFPN
from models.minkloc_multimodal import ResNetFPNv2

from tools.utils import set_seed
set_seed(7)
# from tools.options import Options
# args = Options().parse()

def model_factory(
                fuse_method, cloud_fe_size, image_fe_size, 
                cloud_planes, cloud_layers, cloud_topdown,
                image_useallstages, image_fe, 
                ):



    cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                        planes=cloud_planes, layers=cloud_layers, num_top_down=cloud_topdown,
                        conv0_kernel_size=5, block='ECABasicBlock', pooling_method='GeM')
    




    # image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
    #                     fh_num_bottom_up=4, fh_num_top_down=0,
    #                     add_basicblock=resnetfpn_add_basicblock)
    image_fe = ResNetFPNv2(
                        image_fe=image_fe,
                        image_pool_method='GeM',
                        image_useallstages=image_useallstages,
                        output_dim=image_fe_size,
    )


    model = MinkLocMultimodal(
                            cloud_fe, cloud_fe_size, image_fe, image_fe_size,
                            fuse_method=fuse_method
                            )




    return model
