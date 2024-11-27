import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import saliNet_seg as model
from torch.autograd import Variable
#import med_dataset_wsl_ATLAS_cam as med_dataset_wsl
import med_dataset_wsl
from torchvision.utils import save_image
import datetime
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from loss import BCEDiceLoss

import med_dataset#_wsl as med_dataset
from torchvision.utils import save_image
from PIL import Image
from pylab import *
import imageio

from tensorboardX import SummaryWriter
writer = SummaryWriter("./logs/ATLAS-saliNet_seg")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 100
batch_size = 4
bs = 1
log_path = './checkpoints'

for f in range(3,4):
    learning_rate = 0.0003
    a = 1
    b = 1
    c = 1
    d = 1
    e = 1
    #f = 1
    #folder_name = "{0}_{1}_{2}_{3}_{4}{5}{6}{7}{8}{9}".format('tw',datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),batch_size,learning_rate,a,b,c,d,e,f)
    folder_name = "ATLAS-saliNet_seg"
    #folder_name = './checkpoints/uaeshare_2020-11-12_10:50:26_2_0.0003_1211'
    #格式转换
    
    trans = transforms.ToTensor()

    custom_dataset = med_dataset_wsl.med('./data/ATLAS/train/train',
                                    './data/ATLAS/train/anno',
                                    transform = trans)
    train_dataset = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)#,
                                                #num_workers = 32)

    custom_dataset_val = med_dataset.med('./data/ATLAS/val/train',
                                    './data/ATLAS/val/anno',
                                    transform = trans)
    test_dataset = torch.utils.data.DataLoader(dataset=custom_dataset_val,
                                                batch_size = bs,
                                                shuffle = True)
                                
    def denorm(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)  #clamp，区间限定函数，返回的value介于0,1之间

    def rewrite(x):
        y = (x > 0.5).astype('uint8')
        return y

    def mergeout(x):
        ori_image = x.reshape(x.size(0),1,240,240)
        return ori_image

    def mergepic(x):
        ori_image = x.reshape(x.size(0),1,240,240)
        return ori_image

    def test(pic1,pic2):
        picc1 = pic1#[0,:,:,:]
        picc2 = pic2#[0,:,:,:]
        if picc1.shape[2] == picc2.shape[2]:
            N = picc1.shape[2]

            picc1 = rewrite(picc1)
            picc2 = rewrite(picc2)

            picc1_flat = picc1.reshape(-1,N)
            picc1_flat = picc1_flat.T
            picc2_flat = picc2.reshape(-1,N)
            picc2_flat = picc2_flat.T

            intersection = picc1_flat * picc2_flat
            T = picc2_flat.sum(1)

            P = picc1_flat.sum(1)
            TP = intersection.sum(1)
            FP = P-TP
            FN = T-TP

            dice = ((2*TP) +1) / (T+P+1)
            dice_loss = 1 - dice.sum()/N

            ppv = (TP+1) / (TP+FP+1)
            ppv_loss = 1 - ppv.sum()/N

            sensitivity = (TP+1) / (TP+FN+1)
            sensitivity_loss = 1 - sensitivity.sum()/N

            iou = (TP+1) / (T+P-TP +1)
            iou_loss = 1 - iou.sum()/N

        else:
            print('size dont match')

        return dice,dice_loss,ppv,ppv_loss,sensitivity,sensitivity_loss,iou,iou_loss

    net = model.CPD_ResNet().to(device)
    

    #net = nn.DataParallel(net, device_ids=[0, 1])
    #net.load_state_dict(torch.load('./checkpoints/uaeshare_2020-11-12_10:50:26_2_0.0003_1211/15_wsl_MODEL.pth', map_location='cuda:0'))
    #criterionbce = nn.BCELoss()
    criterionbce = BCEDiceLoss()
    criterionmse = nn.MSELoss()

    # training
    try:
        for epoch in range(num_epochs):

            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            if epoch%3 == 0:
                learning_rate = learning_rate*0.8
                #print(learning_rate)

            time00 = time.time()
            for i, (imagei,imagea,imageal) in enumerate(train_dataset):
                time0 = time.time()
                #imagei = imagei.type(torch.FloatTensor)
                imagei = imagei.to(device)
                #imagea = imagea.type(torch.FloatTensor)
                imagea = imagea.to(device)
                imageal = imageal.to(device)
                #label = label.to(device)

                Map, seg, ProMap = net(imagei)
                
                if epoch%99 ==0:
                #创建文件夹保存运行结果
                    if not os.path.exists(os.path.join(log_path,folder_name,'SEG_GT','epoch-{}'.format(epoch+1))):
                        os.makedirs(os.path.join(log_path,folder_name,'SEG_GT','epoch-{}'.format(epoch+1)))
                        
                    if not os.path.exists(os.path.join(log_path,folder_name,'SEG_OUT','epoch-{}'.format(epoch+1))):
                        os.makedirs(os.path.join(log_path,folder_name,'SEG_OUT','epoch-{}'.format(epoch+1)))
                        
                    if not os.path.exists(os.path.join(log_path,folder_name,'MAP','epoch-{}'.format(epoch+1))):
                        os.makedirs(os.path.join(log_path,folder_name,'MAP','epoch-{}'.format(epoch+1)))
                        
                    if not os.path.exists(os.path.join(log_path,folder_name,'ProMap','epoch-{}'.format(epoch+1))):
                        os.makedirs(os.path.join(log_path,folder_name,'ProMap','epoch-{}'.format(epoch+1)))    

               
                    oriout_images = mergeout(imagea)
                    save_image(oriout_images, os.path.join(log_path,folder_name,'SEG_GT','epoch-{}'.format(epoch+1),'images-{}.png'.format(i+1)))

                    out_images = mergeout(seg)
                    save_image(out_images,os.path.join(log_path,folder_name,'SEG_OUT','epoch-{}'.format(epoch+1),'image-{}.png'.format(i+1)))

                    seg_Map_images = mergeout(Map)
                    save_image(seg_Map_images,os.path.join(log_path,folder_name,'MAP','epoch-{}'.format(epoch+1),'image-{}.png'.format(i+1)))

                    seg_ProMap_images = mergeout(ProMap)
                    save_image(seg_ProMap_images,os.path.join(log_path,folder_name,'ProMap','epoch-{}'.format(epoch+1),'image-{}.png'.format(i+1)))
                    
                loss1 = criterionbce(Map, imagea)
                loss2 = criterionbce(seg, imagea)

                loss = loss1 + loss2

                loss.backward()

                #优化
                optimizer.step()
                optimizer.zero_grad()

                time1 = time.time()
                timed = time1 - time0
                timea = time1 - time00
                
                #输出Epoch、step、loss、time
                if i%1 == 0:
                    time11 = time.time()
                    timedd = time11-time00
                    B = "Epoch[{}/{}],step[{}/{}],Loss:{:.4f}, loss1: {:.6f}, loss2: {:.6f}, timed:{:.4f}, timedd:{:.4f}".format(epoch+1,num_epochs,i+1,len(train_dataset), loss.item(), loss1.item(), loss2.item(), timed, timedd)
                    
                    print(B)
            if epoch%2==0:        
                torch.save(net.state_dict(),os.path.join(log_path,folder_name,'{}_wsl_MODEL.pth'.format(epoch+1)))
            writer.add_scalar('train_loss', loss , epoch)
    except KeyboardInterrupt:
        torch.save(net.state_dict(),os.path.join(log_path,folder_name,'interrupt_wsl_MODEL.pth'.format(epoch+1)))
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
