



import numpy as np


import torch


import open3d
import numpy as np
import open3d as o3d



def viz_lidar_open3d(pointcloud, colors=None, width=None, height=None):

    x = pointcloud[:,0]  # x position of point
    y = pointcloud[:,1]  # y position of point
    z = pointcloud[:,2]  # z position of point

    pcd = open3d.geometry.PointCloud()
    points = np.hstack([x[:,None],y[:,None],z[:,None]])
    points = open3d.utility.Vector3dVector(points)
    pcd.points = points


    if colors is not None:
        pcd.colors = open3d.utility.Vector3dVector(colors)
    


    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])



    if (width is not None) & (height is not None):
        open3d.visualization.draw_geometries([pcd,FOR1], width=width, height=height)
    else:

        open3d.visualization.draw_geometries([pcd,FOR1])






def pc_array_to_voxel(pc_ndarray: np.ndarray):
    """Convert (N,3) np.ndarray point cloud to voxel grid with shape
    `voxel_shape`. Point boundary is determined `VOXEL_LOWER_POINT`
    and `VOXEL_UPPER_POINT`.
    Args:
        pc_ndarray: The input (N,3) np.ndarray point cloud
    Returns:
        pc_tensor: The output point cloud voxel of type torch.Tensor
        [1, voxel_shape[0], voxel_shape[1], voxel_shape[2]]
    """

    voxel_shape = np.array((72, 72, 48), np.int32)  # (x,y,z)

    VOXEL_LOWER_VALUE = np.array((-0.8, -0.4, -0.2), np.float32)
    VOXEL_UPPER_VALUE = np.array((0.8, 0.4, 0.4), np.float32)  # (x,y,z)
    VOXEL_PER_VALUE = voxel_shape / (
        VOXEL_UPPER_VALUE - VOXEL_LOWER_VALUE
    )  # (x,y,z)




    voxel_index = (
        pc_ndarray - np.tile(VOXEL_LOWER_VALUE, (pc_ndarray.shape[0], 1))
    ) * VOXEL_PER_VALUE
    voxel_index = np.round(voxel_index).astype(np.int32)  # raw index

    # filter out out-of-boundary points
    valid_mask = (voxel_index >= np.array([0, 0, 0], np.int32)) & (
        voxel_index < voxel_shape
    )  # only True in (x,y,z) means valid index
    valid_mask = valid_mask[:, 0] & valid_mask[:, 1] & valid_mask[:, 2]
    voxel_index = voxel_index[valid_mask]  # valid voxel index(inside voxel range)

    # deal with voxel according to index
    pc_voxel = np.zeros(voxel_shape, np.float32)  # [72, 72, 48]
    pc_voxel[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = 1.0
    pc_tensor = torch.unsqueeze(torch.from_numpy(pc_voxel), 0)  # insert channel 1
    return pc_tensor




if __name__ == '__main__':




    pc = np.fromfile('/data/sijie/distil/distil_v33/1400505893170765.bin', dtype=np.float64)
    pc = pc.reshape((-1, 3))
    pc = pc.astype(np.float32)


    # viz_lidar_open3d(pc.copy()*100)

    voxel = pc_array_to_voxel(pc)



    voxel = voxel.squeeze(0).numpy()

    voxel_ids = np.where(voxel==1)

    voxel_ids = np.stack(voxel_ids,axis=1) # (N,3)

    

    # viz_lidar_open3d(voxel_ids)

    a=1


