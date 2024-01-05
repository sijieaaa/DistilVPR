# Author: Jacek Komorowski
# Warsaw University of Technology

# Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project
# For information on dataset see: https://github.com/mikacuy/pointnetvlad

import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import MinkowskiEngine as ME
import torchvision.transforms as TVT


import torchvision

import random
from typing import Dict

from tools.utils_adafusion import pc_array_to_voxel

from tools.utils import set_seed
set_seed(7)
from tools.options import Options
args = Options().parse()

DEBUG = False







def project_onto_image(points, P, width=2448, height=2048, color='depth', checkdims=True):
    """Projects 3D points onto a 2D image plane
    Args:
        points (np.ndarray): (N, 3) 3D points in camera coordinate system
        P (np.ndarray): [fx 0 cx 0; 0 fy cy 0; 0 0 1 0; 0 0 0 1] cam projection
        width (int): width of image
        height (int): height of image
        color (str): 'depth' or 'intensity' to pick colors output
    Return:
        uv (np.ndarray): (N, 2) projected u-v pixel locations in an image
        colors (np.ndarray): (N,) a color value for each pixel location in uv.
        mask (np.ndarray): mask to select only points that project onto image.
    """
    if isinstance(points, np.ndarray):
        x = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        x /= x[:, 2:3]
        x[:, 3] = 1
        x = np.matmul(x, P.transpose())
        if checkdims:
            mask = (x[:, 0] >= 0) & \
                    (x[:, 0] <= width - 1) & \
                    (x[:, 1] >= 0) & \
                    (x[:, 1] <= height - 1)
        else:
            mask = np.ones(x.shape[0], dtype=np.bool)
        uv_all = x.copy()[:,:2]
        colors_all = points[:, 2]

    elif isinstance(points, torch.Tensor):
        x = torch.hstack([
            points[:, :3], 
            torch.ones([points.shape[0],1],dtype=points.dtype,device=points.device)
        ]).float()
        x /= x[:, 2:3]
        x[:, 3] = 1
        x = torch.matmul(x, P.transpose(0,1))
        if checkdims:
            mask = (x[:, 0] >= 0) & \
                    (x[:, 0] <= width - 1) & \
                    (x[:, 1] >= 0) & \
                    (x[:, 1] <= height - 1)
        else:
            mask = torch.ones(x.shape[0], dtype=torch.bool, device=points.device)
        uv_all = x[:,:2]
        colors_all = points[:, 2]


    return uv_all, colors_all, mask






def transform_pts_in_camsystem(points, T, in_place=False):
    """Transforms points given a transform, T. x_out = np.matmul(T, x)
    Args:
        T (np.ndarray): 4x4 transformation matrix
        in_place (bool): if True, self.points is updated
    Returns:
        points (np.ndarray): The transformed points
    """
    assert (T.shape[0] == 4 and T.shape[1] == 4)

    if isinstance(points, np.ndarray):
        points = np.copy(points)
        p = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        points[:, :3] = np.matmul(p, T.transpose())[:, :3]

    elif isinstance(points, torch.Tensor):
        p = torch.hstack([
            points[:, :3], 
            torch.ones([points.shape[0],1],dtype=points.dtype,device=points.device)
        ]).float()
        points[:,:3] = torch.matmul(p, T.transpose(0,1))[:,:3]

    return points






def viz_pts_projected_on_image(pts_uv, colors, image, dpi=100):
    '''
    pts_uv: (N,2)
    colors: (N,)
    image: (C,H,W)
    '''
    pts_uv = pts_uv.cpu()
    colors = colors.cpu()
    image = image.cpu()

    image = image[:3]
    # viz projected points  
    _image = image
    _mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
    _std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    _image = _image*_std + _mean
    _image = torchvision.transforms.ToPILImage()(_image)
    _image = np.array(_image)
    # _uv = _uv[_maskselect==1]
    # _colors = _colors[_maskselect==1]
    _uv = pts_uv
    _colors = colors

    plt.figure(dpi=dpi)
    plt.scatter(_uv[:,0], _uv[:,1], c=_colors, s=3)
    plt.imshow(_image)
    plt.show()






class OxfordDataset(Dataset):
    """
    Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project.
    """
    def __init__(self, dataset_path: str, query_filename: str, image_path: str = None,
                 lidar2image_ndx_path: str = None, transform=None, set_transform=None, image_transform=None,
                 use_cloud: bool = True):
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        
        # '/home/sijie/vpr/benchmark_datasets/training_queries_baseline.pickle'  [21711]
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))

        self.image_path = image_path
        self.lidar2image_ndx_path = lidar2image_ndx_path
        self.image_transform = image_transform
        if args.dataset in ['oxford','oxfordadafusion']:
            self.n_points = 4096    # pointclouds in the dataset are downsampled to 4096 points
        elif args.dataset == 'boreas':
            self.n_points = args.n_points_boreas
        else:
            raise Exception


        self.image_ext = '.png'
        self.use_cloud = use_cloud
        print('{} queries in the dataset'.format(len(self)))


        if args.dataset in ['oxford','oxfordadafusion']:
            assert os.path.exists(self.lidar2image_ndx_path)
            # {14005050100:[1532123132,65165165156,16516516516,...],
            #  14005050100:[1532123132,65165165156,16516516516,...],
            #  ...
            # } 40532
            self.lidar2image_ndx = pickle.load(open(self.lidar2image_ndx_path, 'rb'))
        
        elif args.dataset == 'boreas':
            assert os.path.exists(self.lidar2image_ndx_path)
            # {14005050100:[1532123132,65165165156,16516516516,...],
            #  14005050100:[1532123132,65165165156,16516516516,...],
            #  ...
            # } 40532
            self.lidar2image_ndx = pickle.load(open(self.lidar2image_ndx_path, 'rb'))

        a=1


    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):


        filename = self.queries[ndx].rel_scan_filepath
        result = {
            'ndx': ndx,
            'filename': filename,
            }
        if self.use_cloud:

            file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
            query_pc = self.load_pc(file_pathname) # [4096,3]



            # augmentations
            if args.dataset in ['oxford','oxfordadafusion']:
                query_pc = self.set_transform(query_pc) # rotation, flip
            
            if self.transform is not None:
                query_pc = self.transform(query_pc) # jitter, removepoint, removeblock for boreas
            



            # quantization
            if args.dataset in ['oxford','oxfordadafusion']:
                coords, clouds = ME.utils.sparse_quantize(coordinates=query_pc, features=query_pc, quantization_size=args.oxford_quantization_size) # oxford quantization_size=0.01
            elif args.dataset in ['boreas']:
                coords, clouds = ME.utils.sparse_quantize(coordinates=query_pc, features=query_pc, quantization_size=args.boreas_quantization_size) # boreas quantization_size=1




        if self.image_path is not None:
            img = image4lidar(filename, self.image_path, self.image_ext, self.lidar2image_ndx, k=args.image_lidar_k)
            if self.image_transform is not None:
                img = self.image_transform(img)
            result['image'] = img



        if args.dataset in ['oxford','oxfordadafusion']:
            result['coords'] = coords
            result['clouds'] = clouds
            assert len(coords) == len(clouds) # [-1,1]
            # # ---- voxel
            # # viz_lidar_open3d(coords.numpy())
            # voxel_ids = clouds + 1 # [0,2]
            # voxel_ids = voxel_ids * 48 # [0,100]
            # voxel_ids = voxel_ids.int()
            # # viz_lidar_open3d(voxel_ids.numpy())
            # voxels = torch.zeros(100,100,100).float()
            # voxels[voxel_ids[:,0],voxel_ids[:,1],voxel_ids[:,2]] = 1
            # # _voxel_ids = torch.where(voxels>0)
            # # _voxel_ids = torch.stack(_voxel_ids,dim=1)
            # # viz_lidar_open3d(_voxel_ids.numpy())
            # result['voxels'] = voxels

            # ---- voxel
            voxels = pc_array_to_voxel(pc_ndarray=clouds.numpy())
            result['voxels'] = voxels



        elif args.dataset == 'boreas':

            # # P0_camera  T_cameara_lidar_basedon_pose
            # assert 'lidar_1_4096_interval10' in file_pathname
            # P0_camera_path = file_pathname.replace('lidar_1_4096_interval10','P0_camera_interval10').replace('.npy','.txt')
            # P0_camera = np.loadtxt(P0_camera_path)
            # T_camera_lidar_basedon_pose_path = file_pathname.replace('lidar_1_4096_interval10','T_camera_lidar_basedon_pose_interval10').replace('.npy','.txt')
            # T_camera_lidar_basedon_pose = np.loadtxt(T_camera_lidar_basedon_pose_path)


            # _coords_in_camsystem = transform_pts_in_camsystem(coords.float().numpy(), T_camera_lidar_basedon_pose)
            # h, w = img.shape[1:]
            # assert w>h
            # scale_factor = h / 2048 # scale_factor = h/2048 = 0.125 for boreas
            # _uv, _colors, _mask = project_onto_image(_coords_in_camsystem, P0_camera)
            # _uv = _uv * scale_factor # scale_factor = 0.125 for boreas
            # _selected_indices = (coords[:,0]>0) & (coords[:,0]<1000) \
            #         & (coords[:,1]>0) & (coords[:,1]<1000) \
            #         & (coords[:,2]>-10) & (coords[:,2]<50)
            # _maskselect = _mask & _selected_indices.numpy()




            # _uv = torch.tensor(_uv)
            # _colors = torch.tensor(_colors)
            # _maskselect = torch.tensor(_maskselect).int()
            coords = coords
            clouds = clouds
            # P0_camera = torch.tensor(P0_camera).float()
            # T_camera_lidar_basedon_pose = torch.tensor(T_camera_lidar_basedon_pose).float()


            # assert len(_uv) == len(coords)
            # assert len(_colors) == len(coords)
            # assert len(_maskselect) == len(coords)

            assert len(clouds) == len(coords)



            # result['uv'] = _uv
            # result['colors'] = _colors
            # result['maskselect'] = _maskselect
            result['coords'] = coords
            result['clouds'] = clouds
            # result['P0_camera'] = P0_camera
            # result['T_camera_lidar_basedon_pose'] = T_camera_lidar_basedon_pose


            
            # # viz projected points  
            # _image = img
            # _mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
            # _std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
            # _image = _image*_std + _mean
            # _image = torchvision.transforms.ToPILImage()(_image)
            # _image = np.array(_image)
            # _uv = _uv[_maskselect==1]
            # _colors = _colors[_maskselect==1]
            # plt.scatter(_uv[:,0], _uv[:,1], c=_colors, s=3)
            # plt.imshow(_image)
            # plt.show()
            # plt.close()


            # a=sum(_maskselect)
            aa=1

        else:
            raise Exception
        

            

        return result
    









    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives

    def load_pc(self, filename):
        if args.dataset in ['oxford','oxfordadafusion']:
            # Load point cloud, does not apply any transform
            # Returns Nx3 matrix
            file_path = os.path.join(self.dataset_path, filename)
            pc = np.fromfile(file_path, dtype=np.float64)
            # coords are within -1..1 range in each dimension
            assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(file_path)
            pc = np.reshape(pc, (pc.shape[0] // 3, 3))
            pc = torch.tensor(pc, dtype=torch.float)

        elif args.dataset == 'boreas':
            file_path = os.path.join(self.dataset_path, filename)
            pc = np.load(file_path, allow_pickle=True)
            assert pc.shape[0] == self.n_points, "Error in point cloud shape: {}".format(file_path)
            pc = torch.tensor(pc, dtype=torch.float)

        return pc





def ts_from_filename(filename):
    # Extract timestamp (as integer) from the file path/name
    temp = os.path.split(filename)[1]
    lidar_ts = os.path.splitext(temp)[0]        # LiDAR timestamp
    assert lidar_ts.isdigit(), 'Incorrect lidar timestamp: {}'.format(lidar_ts)

    temp = os.path.split(filename)[0]
    temp = os.path.split(temp)[0]
    traversal = os.path.split(temp)[1]
    if args.dataset in ['oxford','oxfordadafusion']:
        assert len(traversal) == 19, 'Incorrect traversal name: {}'.format(traversal)
    elif args.dataset == 'boreas':
        assert len(traversal) == 23, 'Incorrect traversal name: {}'.format(traversal)
        

    return int(lidar_ts), traversal





def image4lidar(filename, image_path, image_ext, lidar2image_ndx, k=1):
    # Return an image corresponding to the given lidar point cloud (given as a path to .bin file)
    # k: Number of closest images to randomly select from
    lidar_ts, traversal = ts_from_filename(filename)
    assert lidar_ts in lidar2image_ndx, 'Unknown lidar timestamp: {}'.format(lidar_ts)

    # Randomly select one of images linked with the point cloud
    if k is None or k > len(lidar2image_ndx[lidar_ts]):
        k = len(lidar2image_ndx[lidar_ts])

    image_ts = random.choice(lidar2image_ndx[lidar_ts][:k])

    if args.dataset in ['oxford','oxfordadafusion']:
        image_file_path = os.path.join(image_path, traversal, str(image_ts) + image_ext)
        #image_file_path = '/media/sf_Datasets/images4lidar/2014-05-19-13-20-57/1400505893134088.png'
        img = Image.open(image_file_path)
    elif args.dataset == 'boreas':
        image_file_path = os.path.join(image_path, traversal, 'camera_lidar_interval10', str(image_ts) + image_ext)
        #image_file_path = '/media/sf_Datasets/images4lidar/2014-05-19-13-20-57/1400505893134088.png'
        img = Image.open(image_file_path)

    return img





class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position
