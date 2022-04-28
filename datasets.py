import sys
sys.path.append('../pyvacy')
import os
import argparse
import math
import numpy as np
import pandas as pd
import pdb
import pickle
from pyvacy import optim, analysis, sampling

#Processing Tools
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.decomposition import PCA

#Deep Learning Tools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,TensorDataset
from torchvision import datasets, transforms
from numpy.random.mtrand import _rand as global_randstate

# Deterministic output
global_randstate.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_datasets(params):
    if params['dataset'] == 'MNIST':
        train_dataset = datasets.MNIST(
            params['data_loc'],
            train=True,
            download=False,
            transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,))
            ])
        )

        test_dataset = datasets.MNIST(
            params['data_loc'],
            train=False,
            download=False,
            transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,))
            ])
        )
        output_dim = 10
        
        return train_dataset, test_dataset, output_dim

    elif params['dataset'] == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(
            params['data_loc'],
            train=True,
            download=False,
            transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,))
            ])
        )

        test_dataset = datasets.FashionMNIST(
            params['data_loc'],
            train=False,
            download=False,
            transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,))
            ])
        )
        output_dim = 10
        return train_dataset, test_dataset, output_dim

    elif params['dataset'] == 'SVHN':
        train_dataset = datasets.SVHN(
            params['data_loc'],
            split='train',
            download=True,
            transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,))
            ])
        )

        test_dataset = datasets.SVHN(
            params['data_loc'],
            split='test',
            download=True,
            transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,))
            ])
        )
        output_dim = 10
        return train_dataset, test_dataset, output_dim

    elif params['dataset'] == 'Adult':
 
        categorical = [1,3,5,6,7,8,9,13]

        def to_one_hot(df, categorical):
            
            target = 14
            data = []
            for i, key in enumerate(df.columns):
                attr = np.array(df[key]).reshape(-1,1)
                
                if i == target:
                    enc = OrdinalEncoder()
                    enc.fit(attr.reshape(-1,1))
                    attr = enc.transform(attr)
                    ctr=0
                    uniq_values=[]

                elif i in categorical:
                    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
                    enc.fit(attr)
                    attr = enc.transform(attr)
                    data.append(attr)
        
                else:
                    reshape_len = len(attr)
                    scaler = MinMaxScaler()
                    scaler.fit(attr)
                    attr = scaler.transform(attr)
                    attr = attr.flatten().reshape(reshape_len, 1)

                data.append(attr)
                
            return data

        data = pd.read_csv(params['data_loc']+'/Adult/adult45k.csv')
        data.dropna(axis=0, inplace=True)

        data_one_hot = to_one_hot(data, categorical)
        data = [torch.tensor(i) for i in data_one_hot]
        data = torch.hstack(data)

        X = data[:,:-1].float()
        y = data[:,-1].long()
 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
        output_dim = 2
        
        return TensorDataset(X_train,y_train), TensorDataset(X_test,y_test), output_dim

    elif params['dataset'] == 'Gisette':

        X = np.load(params['data_loc']+'/Gisette/gisette_processed_x.npy').astype(float)
        y = np.load(params['data_loc']+'/Gisette/gisette_processed_y.npy')
        y = np.where(y==-1, 0, y)

        # transformer = PCA(n_components=1000)
        # X = transformer.fit_transform(X)
        X = torch.Tensor(X).float()
        y = torch.Tensor(y).long()
        #Scaling down to [0,1]
        X = np.true_divide(X, 999)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        output_dim = 2
        return TensorDataset(X_train,y_train), TensorDataset(X_test,y_test), output_dim

    elif params['dataset'] == 'ENRON':

        X = np.load(params['data_loc']+'/ENRON/enron_count_x.npy').astype(float)
        y = np.load(params['data_loc']+'/ENRON/enron_count_y.npy')

        # transformer = PCA(n_components=100)
        # X = transformer.fit_transform(X)

        X = torch.Tensor(X).float()
        y = torch.Tensor(y).long()
        y = y.reshape(-1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        output_dim = 2
        return TensorDataset(X_train,y_train), TensorDataset(X_test,y_test), output_dim