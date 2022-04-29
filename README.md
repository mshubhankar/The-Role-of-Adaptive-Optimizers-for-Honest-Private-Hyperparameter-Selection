# A cost effective Optimizer for Differentially Private Machine Learning
Making DP Optimizers great again!

Prerequisites
===
This project runs on Python 3.6 and necessary installation packages are in requirements.txt
* Run `pip install -r requirements.txt` to install all necessary packages

Conda can be also be used to building environment

Run `conda env create -f=environment.yml
$ conda activate wosm`

Datasets
===

Datasets can be obtained from the following links:
1. [Gisette](https://archive.ics.uci.edu/ml/datasets/Gisette)
2. [ENRON](https://www.cs.cmu.edu/~enron/)


How to run
===
The CONFIG.py can be used to tweak hyperparameters

Run using `python3 main.py [data_location]` 

Figure 1 for LT vs MA comparison graphs can be generated using Figure1.ipynb

Simulation experiments (Figure 2 - 5) can be obtained using the python notebook inside `Simulation/`
The Results folder contains some of our experiment results. These can be used to replicate the graphs.

The DPAdamWOSM optimizer code can be found in `pyvacy/optim/optimizer.py`
