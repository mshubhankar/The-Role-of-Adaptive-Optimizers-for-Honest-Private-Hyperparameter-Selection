#Deep Learning Tools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pyvacy import optim, analysis, sampling
from utils import store_gradients, store_gradients_nonDP, get_optimizer, get_norm

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, input_dim, device='cpu'):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 8, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            Flatten(),
            nn.Linear(288, 10),
            nn.LogSoftmax(dim=1)
        ).to(device)

    def forward(self, x):
        return self.model(x)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(LogisticRegression, self).__init__()
        self.model = nn.Sequential(
                Flatten(),
                nn.Linear(input_dim, output_dim)
            ).to(device)

    def forward(self, x):
        return self.model(x)


class TwoLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(TwoLayer, self).__init__()
        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(input_dim, 100),
            nn.Linear(100, output_dim),
            nn.LogSoftmax(dim=1)
        ).to(device)

    def forward(self, x):
        return self.model(x)