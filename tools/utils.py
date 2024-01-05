# Author: Jacek Komorowski
# Warsaw University of Technology

import os
import configparser
import time
import pickle
import torch 
import numpy as np
import random


import matplotlib.pyplot as plt
import cv2
from tools.options import Options
args = Options().parse()





def get_datetime():
    return time.strftime("%Y%m%d_%H%M")





class ModelParams:
    def __init__(self, model_params_path):
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.mink_quantization_size = params.getfloat('mink_quantization_size', 0.01)

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))

        print('')



class MinkLocParams:
    """
    Params for training MinkLoc models on Oxford dataset
    """
    def __init__(self, params_path, model_params_path=None):
        """
        Configuration files
        :param path: General configuration file
        :param model_params: Model-specific configuration
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']


        if args.dataset in ['oxford', 'oxfordadafusion']:
            self.num_points = params.getint('num_points', 4096)
        elif args.dataset == 'boreas':
            self.num_points = args.n_points_boreas
        else:
            raise Exception




        self.dataset_folder = args.dataset_folder
        self.use_cloud = params.getboolean('use_cloud', True)




        self._check_params()



    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)



    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e not in ['model_params']:
                print('{}: {}'.format(e, param_dict[e]))

        # if self.model_params is not None:
        #     self.model_params.print()
        print('')








def set_seed(seed=7):
    # seed = 7
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(7)





