# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from datasets.oxford import OxfordDataset
from datasets.augmentation import TrainTransform, TrainSetTransform, TrainRGBTransform, ValRGBTransform
from datasets.samplers import BatchSampler
from tools.utils import MinkLocParams
try: from viz_lidar_mayavi_open3d import *
except: None

import matplotlib.pyplot as plt

import torchvision

import os

from datasets.make_collate_fn import make_collate_fn
from datasets.make_collate_fn import make_collate_fn_bak


from tools.options import Options
args = Options().parse()
from tools.utils import set_seed
set_seed(7)










def make_datasets():
    # Create training and validation datasets
    datasets = {}
    # train_transform = TrainTransform(params.aug_mode)
    # train_set_transform = TrainSetTransform(params.aug_mode)
    train_transform = TrainTransform(aug_mode=1)
    train_set_transform = TrainSetTransform(aug_mode=1)


    # if params.use_rgb:
    #     image_train_transform = TrainRGBTransform(aug_mode=1)
    #     image_val_transform = ValRGBTransform()
    # else:
    #     image_train_transform = None
    #     image_val_transform = None
    image_train_transform = TrainRGBTransform(aug_mode=1)
    image_val_transform = ValRGBTransform()


    if args.dataset == 'oxford':
        lidar2image_ndx_path = os.path.join(args.image_path, 'lidar2image_ndx.pickle')

        datasets['train'] = OxfordDataset(
            args.dataset_folder, 
            # query_filename=params.train_file, 
            query_filename='training_queries_baseline.pickle', 
            image_path=args.image_path,
            # lidar2image_ndx_path=params.lidar2image_ndx_path, 
            lidar2image_ndx_path=lidar2image_ndx_path, 
            transform=train_transform,
            set_transform=train_set_transform, 
            image_transform=image_train_transform,
            use_cloud=True
        )



            

    elif args.dataset == 'oxfordadafusion':
        lidar2image_ndx_path = os.path.join(args.image_path, 'oxfordadafusion_lidar2image_ndx.pickle')

        datasets['train'] = OxfordDataset(
            args.dataset_folder, 
            query_filename='oxfordadafusion_training_queries_baseline.pickle', 
            image_path=args.image_path,
            # lidar2image_ndx_path=params.lidar2image_ndx_path, 
            lidar2image_ndx_path=lidar2image_ndx_path, 
            transform=train_transform,
            set_transform=train_set_transform, 
            image_transform=image_train_transform,
            use_cloud=True
        )





    elif args.dataset == 'boreas':
        lidar2image_ndx_path = os.path.join(args.dataset_folder, 'boreas_lidar2image_ndx.pickle')
        assert os.path.exists(lidar2image_ndx_path)

        datasets['train'] = OxfordDataset(
            args.dataset_folder, 
            query_filename='boreas_training_queries_baseline.pickle', 
            image_path=args.image_path,
            # lidar2image_ndx_path=params.lidar2image_ndx_path, 
            lidar2image_ndx_path=lidar2image_ndx_path, 
            transform=train_transform,
            set_transform=train_set_transform, 
            image_transform=image_train_transform,
            use_cloud=True
        )



    else:
        raise Exception


    # a = len(datasets['train'])
    # b = len(datasets['val'])



    return datasets






def make_dataloaders():
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets()




    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], 
                                 batch_size = args.train_batch_size,
                                 batch_size_limit = args.train_batch_size,
                                 batch_expansion_rate = None
                                 )
    
    # ---- for multi-stage training
    train_collate_fn = make_collate_fn(
        datasets['train'],  
    )
    dataloders['train'] = DataLoader(datasets['train'], 
                                     batch_sampler=train_sampler, 
                                     collate_fn=train_collate_fn,
                                     num_workers=args.num_workers, 
                                     pin_memory=True)
    

    

    # ---- for single stage training
    train_preloading_collate_fn = make_collate_fn_bak(
        datasets['train'],
    )
    dataloders['train_preloading'] = DataLoader(
                                                datasets['train'],
                                                batch_size=args.train_batch_split_size, 
                                                collate_fn=train_preloading_collate_fn, 
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    




    return dataloders


