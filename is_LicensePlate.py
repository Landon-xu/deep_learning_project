import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from PIL import Image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets

# Load and normalize dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset =
# trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
# testset =
# testloader =

classes = ('True', 'False')

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
