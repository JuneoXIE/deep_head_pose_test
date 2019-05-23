import sys, os, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import hopenet
import _pickle as pickle
import tensorflow as tf
import PIL
from torchvision import transforms
import utils

# 保存所有sample相对路径的txt文件路径
file_path = r"/xny/deep-head-pose-master/code/meta_test_dataset.txt"

# 保存测试图片的路径
data_dir = r"/xny/deep-head-pose-master/code/test_datasets"

# 保存测试图片bbox的txt文件的路径
bbox_file_path = r"/xny/deep-head-pose-master/code/bboxes.dat"


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines
    
    
def pretreat_img(file_path,data_dir,bbox_file_path):
    img_names = get_list_from_filenames(file_path)
    bbox_file = open(bbox_file_path, 'rb')
    bboxes = np.array(pickle.load(bbox_file))
    bbox_file.close()
    testset = []
    for index, img_name in enumerate(img_names):
        imgpath = data_dir+'/'+img_name
        img = PIL.Image.open(imgpath)
        
        x0 = bboxes[index,0,0] # left
        y0 = bboxes[index,0,1] # upper
        x1 = bboxes[index,0,2] # right
        y1 = bboxes[index,0,3] # lower
        bbox_width = abs(x1- x0)
        bbox_height = abs(y1 - y0)
        x0 -= 3 * bbox_width / 4
        x1 += 3 * bbox_width / 4
        y0 -= 3 * bbox_height / 4
        y1 += bbox_height / 4
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1= min(img.size[1], x1)
        y1 = min(img.size[0], y1)
        # 裁剪坐标为[y0:y1, x0:x1]
        img = img.crop((int(x0), int(y0), int(x1), int(y1)))
        
        transform1 = transforms.Compose([transforms.Scale(224),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img = transform1(img)
        
        img_shape = img.shape
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        
        testset.append(img)
        
    return testset
    
    
if __name__ == '__main__':
    # Load model
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    saved_state_dict = torch.load("hopenet_robust_alpha1.pkl")
    model.load_state_dict(saved_state_dict)
    model.cuda(0)
    
    # Load Testset
    testset = pretreat_img(file_path,data_dir,bbox_file_path)
    
    # Test the Model
    model.eval()  # Change model to 'eval'
    total = 0
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(0)
    

    for index, image in enumerate(testset):
    
        image = Variable(image).cuda(0)
        yaw, pitch, roll = model(image)

        # Binned predictions
        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

        print("Image index:",index)
        print("Yaw:",yaw_predicted)
        print("Pitch:",pitch_predicted)
        print("Roll:",roll_predicted)
        
        if yaw_predicted > 0:
            output_yaw = "face turns left:"
        if yaw_predicted < 0:
            output_yaw = "face turns right:"
        print(output_yaw+str(abs(yaw_predicted))+" degrees")
        
        if pitch_predicted > 0:
            output_pitch = "face upwards:"
        if pitch_predicted < 0:
            output_pitch = "face downwards:"
        print(output_pitch+str(abs(pitch_predicted))+" degrees")
        
        if roll_predicted > 0:
            output_roll = "face bends to the right:"
        if roll_predicted < 0:
            output_roll = "face bends to the left:"
        print(output_roll+str(abs(roll_predicted))+" degrees")
        
        if yaw_predicted==0 and pitch_predicted==0 and roll_predicted==0:
            print("Initial ststus")