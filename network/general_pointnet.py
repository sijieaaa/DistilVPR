
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetSAModule

import torch






class PointNetSimple(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """


    def __init__(self, input_feature_dim=0, use_xyz=True):
        super(PointNetSimple, self).__init__()

        self.sa1 = PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[input_feature_dim, 32, 32],
                use_xyz=use_xyz
        )


        self.sa2 = PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=32,
                mlp=[32, 64, 64],
                use_xyz=use_xyz
        )


        self.sa3 = PointnetSAModule(
                npoint=256,
                radius=0.4,
                nsample=16,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz
        )




    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features



    def forward(self, feed_dict):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """

        clouds = feed_dict['clouds'] # [b, n, 3]


        xyz, features = self._break_up_pc(clouds)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features = self.sa1(xyz, features)

        xyz, features = self.sa2(xyz, features)

        xyz, features = self.sa3(xyz, features)



        return features


if __name__ == '__main__':
    backbone_net = PointNetSimple(input_feature_dim=0).to("cuda")
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(160, 4096, 3).to("cuda"))
    print(out.shape)
