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
        self.img_ilist = img_ilist#[:100]
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
        imgaa = Image.open(img_apath).convert('L')
        imga = imga.resize((240,240))
        imgaa = imgaa.resize((240,240))
        #TODO:
        # imga=np.array(imga)
        ######################################
        ##  Do pic resize or others here     ##
        #imga = np.load(img_apath)
        #print(imga)
        #print(imga.size)
        #######################################

        if self.transform is not None:
            imgi = self.transform(imgi)
            imga = self.transform(imga)
            
            # imga  = torch.from_numpy(imga.astype(np.uint8)) 
            # imga  =  imga.transpose(0,2)
            imgaa = self.transform(imgaa)

            imagei = imgi#.type(torch.FloatTensor)
            imagea = imga#.type(torch.FloatTensor)
            imageaa = imgaa
        #print(imagea.shape)
        return imagei,imagea,imageaa

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.img_ilist)
'''
class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        if not os.path.exists(image_path):
            image_path, label_path = self.data_list[index + 1
                                                    ]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
'''
'''
trans = transforms.ToTensor()
batch_size = 1
custom_dataset = med('/media/hhh/ForUbuntu/Code/Brats_code/Dataset/BRATS_PRE/N_BRATS_HALF1',
                                '/media/hhh/ForUbuntu/Code/Brats_code/Dataset/BRATS_PRE/N_BRATS_HALF1_ANNO',
                                transform = trans)
train_dataset = torch.utils.data.DataLoader(dataset=custom_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)
for i, (imagei,imagea) in enumerate(train_dataset):
    if i<=3:
        print(imagei)
        print(imagea)
        img_i = imagei[0,0,:,:]
        img_a = imagea[0,0,:,:]
        print(img_i.shape)
        print(img_i[150:170,150:170])
        #print(npamg[200:210,200:210])
'''
