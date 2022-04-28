import torch
# File for experiment configuration that is used to setup the for loops for
# mnist_test.py and plot_scripts that work on the results and logs generated
# by running mnist_test.py

# Set RESULTS_DIR to a new folder for a clean experiment
# (or rm -rf the RESULTS_DIR folder, so that previous results don't bash new
# results, since the log files get appended.)

#*****************************************************************************#

RESULTS_DIR='testing'
DENOM_NORM_FNAME="Denom_norm.csv"
TRAIN_LOSS_FNAME="Train_loss.csv"
TEST_LOSS_FNAME="Test_loss.csv"
TEST_ACCURACY_FNAME="Test_accuracy.csv"
# TRACK_ESS_INDICES=[574, 623, 1123, 2324, 2839, 3124, 4235, 5234, 6892, 7120]
TRACK_ESS_INDICES=[1,2,3,4,5,6,7,8,9,10]

params = {
	'delta' : 1e-6,  #delta value for DP
	'device' : 'cuda' if torch.cuda.is_available() else 'cpu', #device for training
	'l2_norm_clip' : 0.5, #clipping threshold for gradients
	'l2_penalty' : 1e-9, #l2 weight penalty for weights
	'lr' : 0.001, #learning rate
	'microbatch_size' : 1, #microbatch size -- sets granularity for DP
	'minibatch_size' : 250, #batchsize for training
	'noise_multiplier' : 2, #epsilon for DP
	'optimizer' : 'DPSGD', #training optimizers -- more options in utils.py
	'data_loc' : 'data', #location to get dataset	
	'oracle' : False, #flag to use raw second moments 
	'architecture' : 'LR', #model architectures -- more options in models.py
	'iterations' : 101, #epochs for training. each epoch = |N|/minibatch iters
	'cutoff' : 0, #cutoff of second moments for DPAdamWOSM
	'get_esslr': False, #set mean of ess to lr at first cutoff of DPAdamWOSM
	'betas' : (0.9, 0.999), #beta values for all Adam optimizers
	'dataset' : 'MNIST', #Dataset for training - MNIST, FashionMNIST, Adult, SVHN
	'seeds' : [1, 2, 3, 4, 5] #Seeds
}

variable_parameters_dict = { # This variable dictionary can used to run batch tasks for multiple hyperparameters
	'l2_norm_clip' : [0.1, 0.2, 0.5, 1],
	'lr' : [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1],
}