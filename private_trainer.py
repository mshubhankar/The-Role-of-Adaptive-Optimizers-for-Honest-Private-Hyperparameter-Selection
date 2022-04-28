import sys
sys.path.append('../pyvacy')
import os
import argparse
import math
import numpy as np
import pdb
import pickle

#Deep Learning Tools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pyvacy import optim, analysis, sampling
from utils import store_gradients, store_gradients_nonDP, get_optimizer, get_norm, store_topk_ess, get_ess, store_ess
from numpy.random.mtrand import _rand as global_randstate

# Local files
import CONFIG
import models
from datasets import load_datasets

# Deterministic output
# global_randstate.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def inv_sqrt(num):
    return (1/(math.sqrt(num)))

def print_DP_details(params):
    eps = analysis.epsilon(
                params['data_length'],
                params['minibatch_size'],
                params['noise_multiplier'],
                params['iterations'],
                params['delta']
            )

    if(params['optimizer'].startswith('DP')):
        print('Achieves ({}, {})-DP with noise {}'.format(
           eps,
            params['delta'],
            params['noise_multiplier']
        ))
    return eps


def train(params):
    curr_dir = params['dir']
    train_dataset, test_dataset, output_dim = load_datasets(params)
    print(params['dataset'])
    # TODO: Would be nice if we could do this classifier and loss function
    # selection in a better fashion?
    params['data_length'] = len(train_dataset)
    # params['iterations'] = int((params['epochs'] * params['data_length']) / params['minibatch_size']) #Need this for IIDBatchSampling 
    
    eps = print_DP_details(params)
    
    params['epsilon'] = eps
    
    dp_log_file = open(params['dir']+"Parameter_details","w")
    for key, value in params.items():
        print(key, ":", value, file=dp_log_file)
    print(params, file=dp_log_file)       
    dp_log_file.close()
    print(params)

    # params['lr'] = get_ess(6, params)
    for seed in params['seeds']:
        
        if(params['architecture']=='LR'):
            classifier = models.LogisticRegression(input_dim=np.prod(train_dataset[0][0].shape),
                            output_dim = output_dim, device = params['device']) 
            loss_function = nn.CrossEntropyLoss()

        elif(params['architecture']=='CNN'):
            classifier = models.Classifier(input_dim=np.prod(train_dataset[0][0].shape),
                            device=params['device'])
            loss_function = nn.NLLLoss()

        elif(params['architecture']=='TLNN'):
            classifier = models.TwoLayer(input_dim=np.prod(train_dataset[0][0].shape),
                            output_dim = output_dim, device=params['device'])
            loss_function = nn.NLLLoss()
            
        log_file = open(curr_dir+'log'+str(seed)+'.txt', 'w')
        global_randstate.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        params['seed'] = seed
        optimizer = get_optimizer(params, classifier)


        all_train_loss = []
        all_test_loss = []
        all_test_accuracy = []
        ind_test_loss=[]
        all_ind_test_loss=[]
        sec_moment=[]
        all_second_moments=[]
        all_denom_norms=[]
        topk_ess=[]
        ess_mean=[]
        ess_std=[]
        tess_indices=[]
        stage_length = 100

        minibatch_loader, microbatch_loader = sampling.get_data_loaders(
            params['minibatch_size'],
            params['microbatch_size'],
            int(params['iterations'])
            )

        iter_ctr = 0
        for X_minibatch, y_minibatch in minibatch_loader(train_dataset):
            
            optimizer.zero_grad()
            all_micro_norms = list()
            minibatch_loss = 0
            for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
                X_microbatch = X_microbatch.to(params['device'])
                y_microbatch = y_microbatch.to(params['device'])


                optimizer.zero_microbatch_grad()
                loss = loss_function(classifier(X_microbatch), y_microbatch)
                loss.backward()
                minibatch_loss += loss.item()
                all_micro_norms.append(get_norm(optimizer.param_groups))
                optimizer.microbatch_step()
     
            store_gradients(all_norms=all_micro_norms, folder=curr_dir)
            
            optimizer.step()
            if(params['optimizer']=='DPAdam'):
                ess_mean.append(optimizer.ess_mean)
                ess_std.append(optimizer.ess_std)
            #print(optimizer.top_effective_ss)

            minibatch_loss /= params['minibatch_size']

            if(iter_ctr%stage_length==0):
                X, y = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset))))
                X, y  = X.to(params['device']), y.to(params['device'])
                y_pred = classifier(X)
                test_loss = loss_function(y_pred,y)
                y_pred = y_pred.max(1)[1]
                count = 0.
                correct = 0.
                incorrect_ind = []
                for pred, actual in zip(y_pred, y):
                    if pred.item() == actual.item():
                        correct += 1.
                    else:
                        incorrect_ind.append(int(count)) #Store indices of incorrect points
                    count += 1.
               
                
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])

                print('[Stage %d/%d] [Training Loss: %f] [Testing Loss %f] [Testing Accuracy %f]' %
                 (iter_ctr/stage_length+1, params['iterations']/stage_length, minibatch_loss, test_loss.item(), correct/count))
                print('[Stage %d/%d] [Training Loss: %f] [Testing Loss %f] [Testing Accuracy %f]' %
                 (iter_ctr/stage_length+1, params['iterations']/stage_length, minibatch_loss, test_loss.item(), correct/count),file=log_file)
                all_train_loss.append(minibatch_loss)
                all_test_loss.append(test_loss.item())
                all_test_accuracy.append(correct/count * 100)

            iter_ctr=iter_ctr+1

            #for param_group in optimizer.param_groups:
                #param_group['lr']=params['lr']*inv_sqrt(epoch+1)
        
        for i in range(len(all_train_loss)):
            with open(curr_dir+CONFIG.TRAIN_LOSS_FNAME+str(seed),'a') as outfile:
                outfile.write(str(i+1)+","+str(all_train_loss[i])+"\n")

        for i in range(len(all_test_loss)):
            with open(curr_dir+CONFIG.TEST_LOSS_FNAME+str(seed),'a') as outfile:
                outfile.write(str(i+1)+","+str(all_test_loss[i])+"\n")

        for i in range(len(all_test_accuracy)):
            with open(curr_dir+CONFIG.TEST_ACCURACY_FNAME+str(seed),'a') as outfile:
                outfile.write(str(i+1)+","+str(all_test_accuracy[i])+"\n")

        if(params['optimizer']=='DPAdam'):
            store_ess(ess_mean,ess_std,folder=curr_dir,seed=seed)
        # torch.save(classifier.state_dict(), curr_dir+'classifier_statedicts.pt')

        #SAVE all second moments
        # np.save(curr_dir+'second_moments.npy', all_second_moments)
    return classifier

