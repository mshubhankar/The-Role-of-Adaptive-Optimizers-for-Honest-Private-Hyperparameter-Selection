# Basic Python imports
import sys
sys.path.append('../pyvacy')
import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pdb
import sys, getopt
# Deep Learning Tools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pyvacy import optim, analysis, sampling
from utils import store_gradients, store_gradients_nonDP, get_optimizer, get_norm
from numpy.random.mtrand import _rand as global_randstate
from sklearn.model_selection import ParameterGrid

# Local Files
import CONFIG
import private_trainer, nonprivate_trainer
import models

# global_randstate.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    params = CONFIG.params
    params['data_loc'] = sys.argv[1]
    
    if not os.path.exists(CONFIG.RESULTS_DIR):
        os.mkdir(CONFIG.RESULTS_DIR)
    
    variables_dict = CONFIG.variable_parameters_dict

    variables = list(ParameterGrid(variables_dict))
    for conf in variables:
        params.update(conf)
        experiment_dir = CONFIG.RESULTS_DIR+"/"+params['optimizer']+"/" 
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)

        newdir = experiment_dir
        for value in conf:
            newdir = newdir + value + '_' + str(conf[value]) +'_'
        newdir += '/'
        params['dir'] = newdir

        if not os.path.exists(newdir):
            os.mkdir(newdir)
        else:
            print("Path Exists Continueing...")
            continue
        if params['optimizer'].startswith('DP') or params['optimizer'].startswith('Clipped'):
            classifier = private_trainer2.train(params)
        else:
            classifier = nonprivate_trainer.train(params)
    
