import torch
import os
import cv2
import numpy as np
from torch.autograd import Variable


def get_file_list(dir, file_list, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            file_list.append(dir)
        else:
            if ext in dir[-3:]:
                file_list.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_file_list(newDir, file_list, ext)

    return file_list


def make_dataset(org_img_folder, target, dataset):
    img_list = get_file_list(org_img_folder, [], 'png')
    for x in range(len(img_list)):
        img_path = img_list[x]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # if len(img.shape) == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_tensor = torch.tensor(img, dtype=torch.int)

        img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=True)  # 增加一个维度

        dataset.append([img_tensor, target])

    # print('Get {} images'.format(len(dataset)))


def start_make_dataset():
    training_set = []
    testing_set = []

    make_dataset('./image_resized/True_resized', 1, training_set)
    make_dataset('./image_resized/False_resized', 0, training_set)
    print('Get {} training images'.format(len(training_set)))

    make_dataset('./image_resized/True_test_resized', 1, testing_set)
    make_dataset('./image_resized/False_test_resized', 0, testing_set)
    print('Get {} testing images'.format(len(testing_set)))

    return training_set, testing_set


# 测试本函数
# start_make_dataset()
