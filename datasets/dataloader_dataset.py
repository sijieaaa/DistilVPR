


import torch.utils.data as data
import MinkowskiEngine as ME
import torch
import os

from datasets.augmentation import ValRGBTransform
import numpy as np
import random

from tools.utils_adafusion import pc_array_to_voxel

from PIL import Image
from datasets.oxford import ts_from_filename
from tools.utils import *
from tools.options import *
args = Options().parse()
set_seed(7)


def image4lidar(filename, image_path, image_ext, lidar2image_ndx, k=None):
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
    
    elif args.dataset == 'boreas':
        image_file_path = os.path.join(image_path, traversal, 'camera_lidar_interval10', str(image_ts) + image_ext)
        
    
    #image_file_path = '/media/sf_Datasets/images4lidar/2014-05-19-13-20-57/1400505893134088.png'
    img = Image.open(image_file_path)
    return img




def load_data_item(file_name, lidar2image_ndx):
    # returns Nx3 matrix
    file_path = os.path.join(args.dataset_folder, file_name)

    result = {}


    if args.dataset in ['oxford','oxfordadafusion']:
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == args.num_points * 3, "Error in point cloud shape: {}".format(file_path)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        result['coords'] = pc
        result['clouds'] = pc.detach().clone()

    elif args.dataset == 'boreas':
        pc = np.load(file_path, allow_pickle=True)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == args.num_points, "Error in point cloud shape: {}".format(file_path)
        pc = torch.tensor(pc, dtype=torch.float)
        result['coords'] = pc
        result['clouds'] = pc.detach().clone()


        P0_camera_path = file_path.replace('lidar_1_4096_interval10','P0_camera_interval10').replace('.npy','.txt')
        P0_camera = np.loadtxt(P0_camera_path)
        T_camera_lidar_basedon_pose_path = file_path.replace('lidar_1_4096_interval10','T_camera_lidar_basedon_pose_interval10').replace('.npy','.txt')
        T_camera_lidar_basedon_pose = np.loadtxt(T_camera_lidar_basedon_pose_path)


        P0_camera = torch.tensor(P0_camera).float()
        T_camera_lidar_basedon_pose = torch.tensor(T_camera_lidar_basedon_pose).float()
        result['P0_camera'] = P0_camera
        result['T_camera_lidar_basedon_pose'] = T_camera_lidar_basedon_pose





    # Get the first closest image for each LiDAR scan
    assert os.path.exists(args.lidar2image_ndx_path), f"Cannot find lidar2image_ndx pickle: {args.lidar2image_ndx_path}"
    # lidar2image_ndx = pickle.load(open(params.lidar2image_ndx_path, 'rb'))
    img = image4lidar(file_name, args.image_path, '.png', lidar2image_ndx, k=1)
    transform = ValRGBTransform()
    # Convert to tensor and normalize
    result['image'] = transform(img)






    return result





def collate_fn(batch_list):

    batch_dict = {}
    coords_list = []
    images_list = []
    clouds_list = []
    voxels_list = []
    # sph_clouds_list = []
    T_camera_lidar_basedon_pose_list = []
    P0_camera_list = []


    for each_batch in batch_list:
        coords = each_batch['coords'] # [4096,3]
        images = each_batch['images']
        clouds = each_batch['clouds']
        voxels = each_batch['voxels']

        
        # if args.dataset == 'boreas':
        #     if args.sph_cloud_fe is not None:
        #         sph_clouds = each_batch['sph_cloud']
        

        if args.dataset in ['oxford','oxfordadafusion']:
            coords = ME.utils.sparse_quantize(coordinates=coords, quantization_size=args.oxford_quantization_size)
        elif args.dataset == 'boreas':
            coords = ME.utils.sparse_quantize(coordinates=coords, quantization_size=args.boreas_quantization_size)



        coords_list.append(coords)
        images_list.append(images)
        clouds_list.append(clouds)
        voxels_list.append(voxels)

        # if args.dataset == 'boreas':
        #     if args.sph_cloud_fe is not None:
        #         sph_clouds_list.append(sph_clouds)
            
        #     T_camera_lidar_basedon_pose = each_batch['T_camera_lidar_basedon_pose']
        #     P0_camera = each_batch['P0_camera']
        #     T_camera_lidar_basedon_pose_list.append(T_camera_lidar_basedon_pose)
        #     P0_camera_list.append(P0_camera)



    coords_list = ME.utils.batched_coordinates(coords_list)
    features_list = torch.ones([len(coords_list), 1])
    images_list = torch.stack(images_list)
    clouds_list = torch.stack(clouds_list)
    voxels_list = torch.stack(voxels_list) # [B, 100, 100, 100]
    # voxels_list = voxels_list.unsqueeze(1) # [B, 1, 100, 100, 100]




    # if args.dataset == 'boreas':
    #     if args.sph_cloud_fe is not None:
    #         sph_clouds_list = torch.stack(sph_clouds_list)

        # T_camera_lidar_basedon_pose_list = torch.stack(T_camera_lidar_basedon_pose_list)
        # P0_camera_list = torch.stack(P0_camera_list)


    if args.dataset == 'boreas':
        # if args.sph_cloud_fe is not None:
        #     batch_dict = {
        #         'coords': coords_list,
        #         'features': features_list,
        #         'images': images_list,
        #         'clouds': clouds_list,
        #         # 'sph_cloud':sph_clouds_list
        #     }
        #     return batch_dict

        
        batch_dict = {
            'coords': coords_list,
            'features': features_list,
            'images': images_list,
            'clouds': clouds_list,
            # 'T_camera_lidar_basedon_pose':T_camera_lidar_basedon_pose_list,
            # 'P0_camera':P0_camera_list
        }
        return batch_dict





    batch_dict = {
        'coords': coords_list,
        'features': features_list,
        'images': images_list,
        'clouds': clouds_list,
        'voxels': voxels_list
    }

    return batch_dict








class DataloaderDataset(data.Dataset):
    def __init__(self, set_dict, device, lidar2image_ndx):

        self.set_dict = set_dict

        # self.params = params
        self.device = device
        self.lidar2image_ndx = lidar2image_ndx
        # self.lidar2image_ndx = pickle.load(open(params.lidar2image_ndx_path, 'rb'))


    def __len__(self):
        length = len(self.set_dict)
        return length


    def __getitem__(self, index):


        data_dict = {}


        x = load_data_item(self.set_dict[index]["query"], self.lidar2image_ndx)


        # quantize in collate_fn
        data_dict['coords'] = x['coords']

        # if args.dataset == 'boreas':
        #     if args.sph_cloud_fe is not None:
        #         data_dict['sph_cloud'] = x['sph_cloud']


        data_dict['images'] = x['image']

        data_dict['clouds'] = x['clouds']


        assert len(x['coords']) == len(x['clouds']) # [-1,1]
        # ---- voxel
        # # viz_lidar_open3d(coords.numpy())
        # voxel_ids = x['clouds'] + 1 # [0,2]
        # voxel_ids = voxel_ids * 48 # [0,100]
        # voxel_ids = voxel_ids.int()
        # # viz_lidar_open3d(voxel_ids.numpy())
        # voxels = torch.zeros(100,100,100).float()
        # voxels[voxel_ids[:,0],voxel_ids[:,1],voxel_ids[:,2]] = 1
        # # _voxel_ids = torch.where(voxels>0)
        # # _voxel_ids = torch.stack(_voxel_ids,dim=1)
        # # viz_lidar_open3d(_voxel_ids.numpy())
        # data_dict['voxels'] = voxels

        # ---- voxel
        voxels = pc_array_to_voxel(x['clouds'].numpy()) # [1,shape0,shape1,shape2] Tensor
        data_dict['voxels'] = voxels
        a=1
    




        # if args.dataset in ['boreas']:
        #     data_dict['P0_camera'] = x['P0_camera']
        #     data_dict['T_camera_lidar_basedon_pose'] = x['T_camera_lidar_basedon_pose']



        return data_dict


