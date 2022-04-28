import sys
sys.path.append('../pyvacy')
import os
import argparse
import math
import numpy as np
import pdb

#Deep Learning Tools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pyvacy import optim, analysis, sampling
from utils import store_gradients, store_gradients_nonDP, get_optimizer, get_norm
from numpy.random.mtrand import _rand as global_randstate

# Local files
import CONFIG
import models
from datasets import load_datasets

# Deterministic output
global_randstate.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def inv_sqrt(num):
    return (1/(math.sqrt(num)))


def train(params):
    curr_dir = params['dir']
    log_file = open(curr_dir+'log.txt', 'w')

    train_dataset, test_dataset, output_dim = load_datasets(params)
    params['microbatch_size']=128
    # TODO: Would be nice if we could do this classifier and loss function
    # selection in a better fashion?
    if(params['architecture']=='LR'):
        classifier = models.LogisticRegression(input_dim=np.prod(train_dataset[0][0].shape),
                        output_dim = output_dim, device = params['device']) 
        loss_function = nn.CrossEntropyLoss()

    elif(params['architecture']=='NN'):
        classifier = models.Classifier(input_dim=np.prod(train_dataset[0][0].shape),
                        device=params['device'])
        loss_function = nn.NLLLoss()

    optimizer = get_optimizer(params, classifier)
   
    print(params['epochs'])
    print(params['minibatch_size'])
    params['iterations'] = int((params['epochs'] * len(train_dataset)) / params['minibatch_size']) #Need this for IIDBatchSampling 

    minibatch_loader, microbatch_loader = sampling.get_data_loaders(
        params['minibatch_size'],
        params['microbatch_size'],
        int(len(train_dataset) / params['minibatch_size'])
        )

    all_train_loss = []
    all_test_loss = []
    all_test_accuracy = []
    all_denom_norms=[]

    print(len(train_dataset))
    print(len(test_dataset))
    print(len(train_dataset)+len(test_dataset))

    for epoch in range(params['epochs']):

        for X_minibatch, y_minibatch in minibatch_loader(train_dataset):            
            all_micro_norms = list()
            minibatch_loss = 0

            for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
                X_microbatch = X_microbatch.to(params['device'])
                y_microbatch = y_microbatch.to(params['device'])
            
                optimizer.zero_grad()
                loss = loss_function(classifier(X_microbatch), y_microbatch)
                loss.backward()

                minibatch_loss += loss.item()
                all_micro_norms.append(get_norm(optimizer.param_groups))
                optimizer.step()

            
            minibatch_loss /= params['minibatch_size']

        X, y = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset))))
        X, y  = X.to(params['device']), y.to(params['device'])

        y_pred = classifier(X)
        test_loss = loss_function(y_pred,y)
        y_pred = y_pred.max(1)[1]
        count = 0.
        correct = 0.
        for pred, actual in zip(y_pred, y):
            if pred.item() == actual.item():
                correct += 1.
            count += 1.
        
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        print('[Epoch %d/%d] [Training Loss: %f] [Testing Loss %f] [Testing Accuracy %f]' %
         (epoch, params['epochs'], minibatch_loss, test_loss.item(), correct/count))
        print('[Epoch %d/%d] [Training Loss: %f] [Testing Loss %f] [Testing Accuracy %f]' %
         (epoch, params['epochs'], minibatch_loss, test_loss.item(), correct/count),file=log_file)
        all_train_loss.append(minibatch_loss)
        all_test_loss.append(test_loss.item())
        all_test_accuracy.append(correct/count * 100)
        
        # for param_group in optimizer.param_groups:
        #     param_group['lr']=params['lr']*inv_sqrt(epoch+1)
    
    for i in range(len(all_train_loss)):
        with open(curr_dir+CONFIG.TRAIN_LOSS_FNAME,'a') as outfile:
            outfile.write(str(i+1)+","+str(all_train_loss[i])+"\n")

    for i in range(len(all_test_loss)):
        with open(curr_dir+CONFIG.TEST_LOSS_FNAME,'a') as outfile:
            outfile.write(str(i+1)+","+str(all_test_loss[i])+"\n")

    for i in range(len(all_denom_norms)):
        with open(curr_dir+CONFIG.DENOM_NORM_FNAME,'a') as outfile:
            outfile.write(str(i+1)+","+str(all_denom_norms[i][0])+"\n")

    for i in range(len(all_test_accuracy)):
        with open(curr_dir+CONFIG.TEST_ACCURACY_FNAME,'a') as outfile:
            outfile.write(str(i+1)+","+str(all_test_accuracy[i])+"\n")

    torch.save(classifier.state_dict(), curr_dir+'classifier_statedicts.pt')

    return classifier