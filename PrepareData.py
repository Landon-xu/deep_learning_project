import torch
import os
import cv2


dataset = []


def make_dataset(org_img_folder, target):
    img_list = get_file_list(org_img_folder, [], 'png')
    for x in range(len(img_list)):
        img_path = img_list[x]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # if len(img.shape) == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_tensor = torch.tensor(img, dtype=torch.int)

        dataset.append([img_tensor, target])


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


make_dataset('./True', 1)
make_dataset('./False', 0)
