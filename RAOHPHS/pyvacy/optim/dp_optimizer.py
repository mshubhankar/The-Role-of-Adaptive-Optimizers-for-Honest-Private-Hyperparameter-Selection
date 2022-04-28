import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adagrad, RMSprop
from pyvacy.optim.optimizer import Adam, AdamWOSM

import pyvacy
import inspect
import math
import pdb
from numpy.random.mtrand import _rand as global_randstate

# Deterministic output
# global_randstate.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, minibatch_size, microbatch_size, l2_norm_clip, noise_multiplier, seed, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)
            global_randstate.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
                group['raw_gradients'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']] 

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad, rg in zip(group['params'], group['accum_grads'], group['raw_gradients']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        
                        rg.zero_()
                        rg.add_(param.grad.data*(self.microbatch_size / self.minibatch_size))

                        noise_matrix = self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data)

                        param.grad.data.add_(noise_matrix)
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            super(DPOptimizerClass, self).step(*args, **kwargs)
    return DPOptimizerClass

AdamClipped = make_optimizer_class(Adam)
DPAdam = make_optimizer_class(Adam)
DPAdagrad = make_optimizer_class(Adagrad)
DPSGD = make_optimizer_class(SGD)
DPRMSprop = make_optimizer_class(RMSprop)
DPAdamWOSM = make_optimizer_class(AdamWOSM)

#SGDNesterov = make_optimizer_class(SGD(nesterov=True))
