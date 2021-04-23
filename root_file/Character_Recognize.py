# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import torch.nn as nn

import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import cv2
from PIL import Image, ImageDraw, ImageFont

import os, random, math, shutil

epochs = 20
learning_rate = 0.002005
# Convert picture format
transform = transforms.Compose([  # transforms.Resize(32,32), # Image resizing
    transforms.ToTensor(),  # Data type adjustment
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # Normalization
)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
           'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
           'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')


# Load data set Here defines a class, related to the data set, the loading of the test set
class Setloader():
    def __init__(self):
        pass

    def trainset_loader(self):
        path = 'data'
        trainset = datasets.ImageFolder(root=path, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
        return trainloader


# Define the class of the network model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)  # 输入颜色通道 ：1  输出通道：6，卷积核：5*5  卷积核默认步长是1  30*30
        self.conv2 = nn.Conv2d(8, 16, 2)  # 这是第二个卷积层，输入通道是：6 ，输出通道：16 ，卷积核：3*3   30*30
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层   10*10
        # self.conv3 = nn.Conv2d(16 , 8 , 2)#第三个卷积层，输入层的输入通道要和上一层传下来的通道一样，这里给了填充是2，这个参数默认是：0，填充可以把边缘信息、特征提取出来，不流失   14*14
        self.fc1 = nn.Linear(16 * 4 * 4, 110)
        self.fc2 = nn.Linear(110, 200)
        self.fc3 = nn.Linear(200, 130)
        self.fc4 = nn.Linear(130, 34)  # Three fully connected layers 65 is the final output has 65 categories

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = F.relu(self.conv3(x))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Train the network model and save the class of model parameters, train, test, and display


class Trainandsave():
    def __init__(self):
        self.net = Net()
        pass

    def train_net(self):
        self.net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(Setloader.trainset_loader(self), 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = self.net(inputs)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:
                    print('[%d ,%5d]loss:%.3f' % (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0
        print('finished training!')
        PATH = './model/net.pth'
        torch.save(self.net.state_dict(), PATH)

    def load_test(self, img):
        self.net.load_state_dict(torch.load('./model/net.pth'))
        image_size = (20, 20)
        rimg = transforms.Resize(image_size)
        loader = transforms.Compose([rimg, transforms.ToTensor()])
        image_tensor = loader(img.convert('RGB')).float()
        tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float(), requires_grad=False)
        oputs = self.net(tensor)
        # print(oputs)
        _, predicted = torch.max(oputs.data, 1)  # 确定一行中最大的值的索引  torch.max(input, dim)
        # print(predicted)
        print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(1)))
        return classes[predicted]

    def imshow(self, img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


class Car_pai():
    def __init__(self):
        imgs1 = Image.open('2.jpg')
        imgs2 = Image.open('3.jpg')
        imgs3 = Image.open('4.jpg')
        imgs4 = Image.open('5.jpg')
        imgs5 = Image.open('6.jpg')
        imgs6 = Image.open('7.png')
        imges = [imgs1, imgs2, imgs3, imgs4, imgs5, imgs6]
        str1 = ['', '', '', '', '', '']
        for i in range(len(imges)):
            # cv2.imwrite('G:\deep_learming\pytorch\data\%d.jpg'%i , imges[i])
            train = Trainandsave()
            str1[i] = train.load_test(imges[i])
            # print(self.str1[i])

        for i in range(len(imges)):
            plt.subplot(1, 8, i + 1), plt.imshow(imges[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
        result = str1[0] + str1[1] + str1[2] + str1[3] + str1[4] + str1[5]
        print('result：' + result)

    # set_split(path)
    # trainloader=Setloader()
    # trainloader = trainloader.trainset_loader()


shishi = Car_pai()  # 车牌识别定位分割
# img = cv2.imread('G:\deep_learming\pytorch\data\jk16.jpg',1)
#
# train = Trainandsave()
# train.train_net()  #训练网络，保存参数
# train.load_test(img)
#
