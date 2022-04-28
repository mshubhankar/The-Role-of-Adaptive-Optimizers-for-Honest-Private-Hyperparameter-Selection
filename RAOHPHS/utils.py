import os
#Deep Learning Tools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pyvacy import optim, analysis, sampling
import numpy as np
import pdb
import csv
import math

def get_ess(epochs, params):
    #Returns the ESS at the end of input argument epochs
    UPDATE_NORM = 0.1
    LEARNING_RATE = 0.001
    exp_avg = torch.Tensor([0])
    exp_avg_sq = torch.Tensor([0])
    step = 0
    eps = 1e-8
    for e in range(epochs):
        for i in range(int(params['data_length']/params['minibatch_size'])):
            # Sample a new noise_matrix
            noise_matrix = params['l2_norm_clip'] * params['noise_multiplier'] 
           
            # Sample a new signal matrix: Generate gradient updates based on update_norm
            # No need to scale signal_matrix by MICRO/MINI since that would be MINI * signal_matrix / MINI
            # So just use one signal_matrix sample instead
            clip_coef = min(params['l2_norm_clip']/ (UPDATE_NORM + 1e-6), 1.)
            signal_matrix =  torch.randn_like(exp_avg) * UPDATE_NORM 
            signal_matrix.mul_(clip_coef)

            # Gradient update
            gradient_update = signal_matrix.add(noise_matrix) 
            gradient_update.mul_(params['microbatch_size']/params['minibatch_size'])

            bias_correction1 = 1 - params['betas'][0] ** step
            bias_correction2 = 1 - params['betas'][1] ** step
           
            exp_avg.mul_(params['betas'][0]).add(gradient_update, alpha=1 - params['betas'][0])
            exp_avg_sq.mul_(params['betas'][1]).addcmul_(gradient_update, gradient_update, value=1-params['betas'][1])

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            ones = torch.ones_like(denom)
            ones.mul_(LEARNING_RATE)
            ess = torch.true_divide(ones, denom)
            step=step+1

    return ess.item() 


def get_norm(param_group):
    norms = list()
    for group in param_group:
            for i in range(len(group['params'])):
                if group['params'][i].requires_grad:
                    gradients = group['params'][i].grad.data
                    norms.append(gradients.norm().item())
    return norms


def store_gradients_nonDP(param_group, folder):
    #Extract layers from model ( Can we ignore Flatten layers somehow?)
    layer_norms = list()
    for group in param_group:
            for i in range(len(group['params'])):
                if group['params'][i].requires_grad:
                    gradients = group['params'][i].grad.data
                    layer_norms.append(gradients.norm().item())

    grad_folder = folder+'gradients/'
    if not os.path.exists(grad_folder):
        os.mkdir(grad_folder)
        for i in range(len(layer_norms)):
            print("Minibatch Norms", file=open(grad_folder+'gradient_norms_L'+str(i)+'.txt', "a"))

    for i in range(len(layer_norms)):
        print(layer_norms[i], file=open(grad_folder+'gradient_norms_L'+str(i)+'.txt', "a"))

def store_ess(ess_mean, ess_std, folder, seed):
    mean = np.array(ess_mean)
    std = np.array(ess_std)
    layers = mean.shape[1]

    ess_folder = folder + 'ess'+str(seed)+'/'
    if not os.path.exists(ess_folder):
        os.mkdir(ess_folder)
    for i in range(layers):
        print("MEAN STD", file=open(ess_folder+'ESS_'+str(i)+'.txt', "a"))
    for i in range(layers):
        for j in range(len(mean)):
            print('%f %f'%(mean[j,i], std[j,i]), file=open(ess_folder+'ESS_'+str(i)+'.txt', "a"))

def store_gradients(all_norms, folder):
    all_norms = np.array(all_norms)
    layers = all_norms.shape[1]
    grad_folder = folder+'gradients/'

    if not os.path.exists(grad_folder):
        os.mkdir(grad_folder)
        for i in range(layers):
            print("MEAN VARIANCE 25th 75th 95th", file=open(grad_folder+'gradient_norm_matrix'+str(i)+'.txt', "a"))
    
    for i in range(layers):
        layer_norms = all_norms[:,i]
        print("%f %f %f %f %f"%(layer_norms.mean(), layer_norms.var(), np.percentile(layer_norms,25),  np.percentile(layer_norms,75), np.percentile(layer_norms,95)), file=open(grad_folder+'gradient_norm_matrix'+str(i)+'.txt', "a"))


def store_topk_ess(index_list, all_topk_ess, ess_mean, ess_std, folder):
    layers = len(all_topk_ess[0])
    print("Num layers of ESS = %d"%(layers))
    if not os.path.exists(folder):
        os.mkdir(folder)
        for i in range(layers):
            print(folder)
            il = index_list[0][i]
            with open('./'+folder+'top_ess_'+str(i)+'.csv','w') as ess_file:
                wr = csv.writer(ess_file, quoting=csv.QUOTE_NONE, escapechar='n')
                # Log the indices of the parameters
                wr.writerow(il)

                # Log the values of the effective step sizes for these indices over all the iterations
                for line in all_topk_ess:
                    wr.writerow(line[i])

            with open('./'+folder+'ess_mean_std'+str(i)+'.csv','w') as ess_file:
                wr = csv.writer(ess_file, quoting=csv.QUOTE_NONE, escapechar='n')
                wr.writerow('MEAN\tSTD')
                # Log the values of the effective step sizes for these indices over all the iterations
                for mean,std in zip(ess_mean, ess_std):
                    wr.writerow('%.3f\t%.3f'%(mean[i],std[i]))

def get_optimizer(params, classifier):

    eps=1e-8
    opt_name = params['optimizer']
    optimizer = None
    if(opt_name == 'DPSGD'):
        optimizer = optim.DPSGD(
            l2_norm_clip=params['l2_norm_clip'],
            noise_multiplier=params['noise_multiplier'],
            minibatch_size=params['minibatch_size'],
            microbatch_size=params['microbatch_size'],
            params=classifier.parameters(),
            lr=params['lr'],
            seed=params['seed']
        )
    if(opt_name == 'DPAdam'):
        optimizer = optim.DPAdam(
                l2_norm_clip=params['l2_norm_clip'],
                noise_multiplier=params['noise_multiplier'],
                minibatch_size=params['minibatch_size'],
                microbatch_size=params['microbatch_size'],
                params=classifier.parameters(),
                lr=params['lr'],
                weight_decay=params['l2_penalty'],
                oracle=params['oracle'],
                eps=eps,
                betas=params['betas'],
                seed=params['seed']               
            )
    if(opt_name == 'DPAdagrad'):
        optimizer = optim.DPAdagrad(
                l2_norm_clip=params['l2_norm_clip'],
                noise_multiplier=params['noise_multiplier'],
                minibatch_size=params['minibatch_size'],
                microbatch_size=params['microbatch_size'],
                params=classifier.parameters(),
                lr=params['lr'],
                weight_decay=params['l2_penalty'],
                seed=params['seed']
            )
    if(opt_name == 'DPRMSProp'):
        optimizer = optim.DPRMSProp(
                    l2_norm_clip=params['l2_norm_clip'],
                    noise_multiplier=params['noise_multiplier'],
                    minibatch_size=params['minibatch_size'],
                    microbatch_size=params['microbatch_size'],
                    params=classifier.parameters(),
                    lr=params['lr'],
                    weight_decay=params['l2_penalty'],
                    seed=params['seed']
                )

    if(opt_name == 'DPSGDNesterov'):
        optimizer = optim.DPSGD(
            l2_norm_clip=params['l2_norm_clip'],
            noise_multiplier=params['noise_multiplier'],
            minibatch_size=params['minibatch_size'],
            microbatch_size=params['microbatch_size'],
            params=classifier.parameters(),
            momentum=0.9,
            weight_decay=params['l2_penalty'],
            lr=params['lr'],
            nesterov=True,
            seed=params['seed']
        )
    if(opt_name == 'DPAdamWOSM'):
        optimizer = optim.DPAdamWOSM(
            l2_norm_clip=params['l2_norm_clip'],
            noise_multiplier=params['noise_multiplier'],
            minibatch_size=params['minibatch_size'],
            microbatch_size=params['microbatch_size'],
            params=classifier.parameters(),
            lr=params['lr'] / (params['noise_multiplier'] * params['l2_norm_clip'] * (params['microbatch_size']/params['minibatch_size']) + 1e-8),  #alpha / (noise_multiplier * clip * (microbatch/minibatch) + e)
            weight_decay=params['l2_penalty'],
            oracle=params['oracle'],
            eps=eps,
            betas=params['betas'],
            cutoff=params['cutoff'],
            get_esslr=params['get_esslr'],
            seed=params['seed']
        )

#Non-Private Optimizers

    if(opt_name == 'Adam'):
        optimizer = torch.optim.Adam(
            params=classifier.parameters(),
            lr=params['lr'],
            weight_decay=params['l2_penalty'],
            # oracle=params['oracle'],
        )
    if(opt_name == 'ClippedAdam'):
        optimizer = optim.AdamClipped(
            l2_norm_clip=params['l2_norm_clip'],
            noise_multiplier=0,
            minibatch_size=params['minibatch_size'],
            microbatch_size=params['microbatch_size'],
            params=classifier.parameters(),
            lr=params['lr'],
            weight_decay=params['l2_penalty'],
            eps=eps,
        )

    if(opt_name == 'SGD'):
        optimizer = torch.optim.SGD(
            params=classifier.parameters(),
            lr=params['lr'],
            weight_decay=params['l2_penalty']
        )

    if(opt_name == 'SGDNesterov'):
        optimizer = torch.optim.SGD(
            params=classifier.parameters(),
            momentum=0.9,
            lr=params['lr'],
            nesterov=True
        )

    if(opt_name == 'Adagrad'):
        optimizer = torch.optim.Adagrad(
            params = classifier.parameters(),
            lr = params['lr'],
            weight_decay=params['l2_penalty']
        )
    return optimizer
