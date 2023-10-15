import os
import cv2
import torch
import numpy as np
import trimesh
from torch.utils.data import Dataset, DataLoader


class TrainData(Dataset):

    def __init__(self, root, train=True, transform=None):
        super(TrainData, self).__init__()
        self.train = train
        self.transform = transform

        if self.train:
            filename = root + 'train.csv'
        else:
            filename = root + 'test.csv'
        ## filename = os.path.join(args.dataset_dir, args.index_file)
        print("From file: {}".format(filename))

        with open(filename, "r") as f:
            self.image_lists = f.readlines()

        self.color_f = []
        self.depth_f = []
        self.depth_b_smpl = []
        train_len = len(self.image_lists)
        print(train_len)
        for i in range(train_len):
            image_names = self.image_lists[i].strip().split(',')
            self.color_f.append(image_names[0])
            self.depth_f.append(image_names[1])
            self.depth_b_smpl.append(image_names[2])

    def __getitem__(self, index):
        color_f_n = self.color_f[index]
        depth_f_n = self.depth_f[index]
        depth_b_smpl_n = self.depth_b_smpl[index]
        color_f = cv2.imread(color_f_n, -1)
        color_f = cv2.cvtColor(color_f, cv2.COLOR_BGR2RGB)
        color_f = self.transform(color_f.astype(np.float32))
        depth_f = cv2.imread(depth_f_n, -1)
        depth_f = self.transform(depth_f.astype(np.float32))
        depth_b_smpl = cv2.imread(depth_b_smpl_n, -1)
        depth_b_smpl = self.transform(depth_b_smpl.astype(np.float32))

        return color_f, depth_f, depth_b_smpl

    def __len__(self):
        return len(self.image_lists)


