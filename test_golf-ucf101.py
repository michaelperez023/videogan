"""
Author: Michael Perez
CAP6610 Term Project
April 2022

References:
https://github.com/cvondrick/videogan
https://github.com/GV1028/videogan
https://github.com/yhjo09/videogan-pytorch
https://github.com/Zasder3/Latent-Neural-Differential-Equations-for-Video-Generation
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import scipy.misc
import numpy as np
import glob
from utils import *
import sys
from argparse import ArgumentParser
from datetime import datetime

from torchvision.datasets import UCF101
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score

parser = ArgumentParser()
parser.add_argument(
    "-d", help="The dimension of each video, must be of shape [3,32,64,64]",
    nargs='*', default=[3,32,64,64]
)
parser.add_argument(
    "-zd", help="The dimension of latent vector [100]",
    type=int, default=100
)
parser.add_argument(
    "-nb", help="The size of batch images [64]",
    type=int, default=10
)
parser.add_argument(
    "-c", help="The checkpoint file name",
    type=str, default="2022-04-24-11-34-57_9_30187"
)
parser.add_argument(
    "-e", help="epochs",
    type=int, default=24
)
parser.add_argument(
    "-se", help="start epoch",
    type=int, default=0
)
args = parser.parse_args()


class Generator(torch.nn.Module):
    def __init__(self, zdim=args.zd):
        super(Generator, self).__init__()
        
        self.zdim = zdim
        
        # Background
        self.conv1b = nn.ConvTranspose2d(zdim, 512, [4,4], [1,1])
        self.bn1b = nn.BatchNorm2d(512)

        self.conv2b = nn.ConvTranspose2d(512, 256, [4,4], [2,2], [1,1])
        self.bn2b = nn.BatchNorm2d(256)

        self.conv3b = nn.ConvTranspose2d(256, 128, [4,4], [2,2], [1,1])
        self.bn3b = nn.BatchNorm2d(128)

        self.conv4b = nn.ConvTranspose2d(128, 64, [4,4], [2,2], [1,1])
        self.bn4b = nn.BatchNorm2d(64)

        self.conv5b = nn.ConvTranspose2d(64, 3, [4,4], [2,2], [1,1])

        # Foreground
        self.conv1 = nn.ConvTranspose3d(zdim, 512, [2,4,4], [1,1,1])
        self.bn1 = nn.BatchNorm3d(512)

        self.conv2 = nn.ConvTranspose3d(512, 256, [4,4,4], [2,2,2], [1,1,1])
        self.bn2 = nn.BatchNorm3d(256)

        self.conv3 = nn.ConvTranspose3d(256, 128, [4,4,4], [2,2,2], [1,1,1])
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.ConvTranspose3d(128, 64, [4,4,4], [2,2,2], [1,1,1])
        self.bn4 = nn.BatchNorm3d(64)

        self.conv5 = nn.ConvTranspose3d(64, 3, [4,4,4], [2,2,2], [1,1,1])

        # Mask
        self.conv5m = nn.ConvTranspose3d(64, 1, [4,4,4], [2,2,2], [1,1,1])

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif classname.lower().find('bn') != -1:
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # Background
        b = F.relu(self.bn1b(self.conv1b(z.unsqueeze(2).unsqueeze(3))))
        b = F.relu(self.bn2b(self.conv2b(b)))
        b = F.relu(self.bn3b(self.conv3b(b)))
        b = F.relu(self.bn4b(self.conv4b(b)))
        b = torch.tanh(self.conv5b(b)).unsqueeze(2)  # b, 3, 1, 64, 64

        # Foreground
        f = F.relu(self.bn1(self.conv1(z.unsqueeze(2).unsqueeze(3).unsqueeze(4))))
        f = F.relu(self.bn2(self.conv2(f)))
        f = F.relu(self.bn3(self.conv3(f)))
        f = F.relu(self.bn4(self.conv4(f)))
        m = torch.sigmoid(self.conv5m(f))   # b, 1, 32, 64, 64
        f = torch.tanh(self.conv5(f))   # b, 3, 32, 64, 64
        
        out = m*f + (1-m)*b

        return out, f, b, m


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 64, [4,4,4], [2,2,2], [1,1,1])

        self.conv2 = nn.Conv3d(64, 128, [4,4,4], [2,2,2], [1,1,1])
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, [4,4,4], [2,2,2], [1,1,1])
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(256, 512, [4,4,4], [2,2,2], [1,1,1])
        self.bn4 = nn.BatchNorm3d(512)

        self.conv5 = nn.Conv3d(512, 1, [2,4,4], [1,1,1])

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif classname.lower().find('bn') != -1:
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.conv5(x)

        return x

ucf_data_dir = "/blue/ctolerfranklin/michaelperez012/videogan-pytorch/UCF-101-partitioned"
ucf_label_dir = "/blue/ctolerfranklin/michaelperez012/videogan-pytorch/ucfTrainTestlist"
frames_per_clip = 32

tfs = transforms.Compose([
            # scale in [0, 1] of type float
            transforms.Lambda(lambda x: x / 255.),
            # reshape into (T, C, H, W)
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            # rescale to 64x64
            transforms.Lambda(lambda x: nn.functional.interpolate(x, (64, 64))),
])

train_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip, train=True, transform=tfs)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.nb, shuffle=True)

def trainDataGen():
    while True:
        for d in train_loader:
            yield d

traindg = trainDataGen()

test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip, train=False, transform=tfs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.nb, shuffle=True)

def testDataGen():
    while True:
        for d in test_loader:
            yield d

testdg = testDataGen()


def main():
    if not os.path.exists("./checkpoints-ft-all"):
        os.makedirs("./checkpoints-ft-all")
    if not os.path.exists("./genvideos"):
        os.makedirs("./genvideos")


    # Model def
    D = Discriminator().cuda()

    params_D = list(filter(lambda p: p.requires_grad, D.parameters()))
    optimizer_D = optim.Adam(params_D, lr=0.0002, betas=(0.5,0.999))

    # Load pretrained
    if args.c is not None:
        state_dicts = torch.load("./checkpoints/{}_state_normal.ckpt".format(args.c))
        D.load_state_dict(state_dicts['model_state_dict'][1], strict=True)
        optimizer_D.load_state_dict(state_dicts['optimizer_state_dict'][1])
        print("Model restored")
        
    D.conv5 = nn.Conv3d(512, 101, [2,4,4], [1,1,1]).cuda() #change last layer of D to have 101 nodes instead of one, to classify all UCF101 videos
    nn.init.normal_(D.conv5.weight, mean=0, std=0.01)
    nn.init.constant_(D.conv5.bias, 0)
        
    print(D)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    criterion = nn.CrossEntropyLoss()
    
    #fine-tune/train
    for epoch in tqdm(range(args.se, args.se + args.e)):
        print("epoch: " + str(args.e))
        optimizer_D.zero_grad()

        realLabelList = next(traindg)
        real, labels = realLabelList[0].cuda(), realLabelList[2]
        
        real = torch.add(torch.multiply(real, 2), -1) # make it [0, 1] -> [-1, 1]
        real = real.to(dtype=torch.float32).transpose(1, 2)
        #print(labels.numpy())
        #print(labels.shape)

        #print(real.shape)
        pr = D(real)

        labels = torch.Tensor(labels.type(dtype=torch.float)).to(device)
        labels = labels.reshape([10, 1, 1, 1])

        dis_loss = criterion(pr, labels.type(dtype=torch.long))

        print("loss: " + str(dis_loss))
        dis_loss.backward()
        optimizer_D.step()

    labelList = next(testdg)
    vids, labels = labelList[0].cuda(), labelList[2]
    
    vids = torch.add(torch.multiply(vids, 2), -1) # make it [0, 1] -> [-1, 1]
    vids = vids.to(dtype=torch.float32).transpose(1, 2)

    result = D(vids).reshape([10, 101]) # reshape: [10, 101, 1, 1, 1] to [10, 101]
    print("results: " + str(result))
    
    result = torch.argmax(result, dim=1).tolist() # get argmax of each row
    
    print("true labels: " + str(labels.numpy()))
    print("predicted labels: " + str(result))
    print("epoch " + str(epoch) + " accuracy: " + str(accuracy_score(labels, result)))

    torch.save({'epoch': epoch,
                    'model_state_dict': [D.state_dict()],
                    'optimizer_state_dict': [optimizer_D.state_dict()]},
                f'checkpoints-ft-all/finetune_state_normal{epoch}.ckpt')
    
if __name__ == '__main__':
    main()
