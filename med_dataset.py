import numpy as np
import torch
import torch.utils.data
import os
import torchvision
from PIL import Image
from torchvision import transforms

import cv2

class med(torch.utils.data.Dataset):
    def __init__(self,img_dir,anno_dir,transform = None):
        # TODO
        # 1. Initialize file paths or a list of file names.
        img_ilist = []
        img_alist = []
        temp = []
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        for subdir in os.listdir(img_dir):
            for file in os.listdir(os.path.join(img_dir,subdir)):
                img_ilist.append(os.path.join(img_dir,subdir,file))
                img_alist.append(os.path.join(anno_dir,subdir,file))
        self.img_ilist = img_ilist
        self.img_alist = img_alist
        self.transform = transform

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        img_ilist = self.img_ilist
        img_alist = self.img_alist
        img_ipath = self.img_ilist[index]
        imgi = Image.open(img_ipath).convert('L')
        #imgi = cv2.imread(img_ipath)
        imgi = imgi.resize((240,240))
        #######################################
        ##  Do pic resize or others here     ##
        #imgi = np.load(img_ipath)
        #print(img_ipath)
        #print(imgi.shape)
        #print(imgi)
        #npy = np.zeros((240,240,1))
        #npy[:,:,0] = imgi
        #print(npy.shape)
        #######################################

        img_apath = self.img_alist[index]
        imga = Image.open(img_apath).convert('L')
        #print(imga.mode)
        #imga = cv2.imread(img_apath)
        imga = imga.resize((240,240))

        #######################################
        ##  Do pic resize or others here     ##
        #imga = np.load(img_apath)
        #print(imga)
        #print(imga.size)
        #######################################

        if self.transform is not None:
             imgi = self.transform(imgi)
             imga = self.transform(imga)
             imagei = imgi#.type(torch.FloatTensor)
             imagea = imga#.type(torch.FloatTensor)

        return imagei,imagea

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.img_ilist)
