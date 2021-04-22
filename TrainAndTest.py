import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import cv2
from torch.autograd import Variable
from PrepareData import start_make_dataset
from TrueOrFalseCNN import TOFCNN


def train():
    BATCH_SIZE = 32
    train_data, test_data = start_make_dataset()
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    saving_path = './model/TrueOrFalseCNN.pth'  # Where to find the saved model
    model = TOFCNN()
    learning_rate = 0.001  # Set the learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Select ADAM as the optimizer
    loss_func = nn.CrossEntropyLoss()  # Select CrossEntropyLoss as the loss function
