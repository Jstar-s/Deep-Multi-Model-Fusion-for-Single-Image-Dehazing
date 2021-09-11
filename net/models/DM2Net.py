import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from option import opt, model_name, log_dir
'''
特征提取网络
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResNext(nn.Module):
    def __init__(self):
        super(ResNext, self).__init__()
        self.model_ft = models.resnext50_32x4d(pretrained=True)
        # self.bn_1 = nn.BatchNorm2d(256)
        self.upsample_1 = nn.UpsamplingNearest2d(scale_factor=4)
        self.conv1_1x1 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.upsample_2 = nn.UpsamplingNearest2d(scale_factor=8)
        self.conv2_1x1 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.upsample_3 = nn.UpsamplingNearest2d(scale_factor=16)
        self.conv3_1x1 = nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.upsample_4 = nn.UpsamplingNearest2d(scale_factor=32)
        self.conv4_1x1 = nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        model_fit = self.model_ft
        # del model_fit.fc, model_fit.avgpool
        conv1 = model_fit.conv1
        bn1 = model_fit.bn1
        relu = model_fit.relu
        maxpool = model_fit.maxpool
        layer1 = model_fit.layer1
        layer2 = model_fit.layer2
        layer3 = model_fit.layer3
        layer4 = model_fit.layer4
        x = conv1(x)
        x = bn1(x)
        x = relu(x)
        x = maxpool(x)
        out1 = layer1(x)
        # print(out1.shape)
        out2 = layer2(out1)
        # print(out2.shape)
        out3 = layer3(out2)
        # print(out3.shape)
        out4 = layer4(out3)
        # print(out4.shape)


        out1 = self.upsample_1(out1)
        out1 = self.conv1_1x1(out1)
        # print("1:", out1.shape)

        out2 = self.upsample_2(out2)
        out2 = self.conv2_1x1(out2)
        # print("2", out2.shape)

        out3 = self.upsample_3(out3)
        out3 = self.conv3_1x1(out3)
        # print("3",out3.shape)

        out4 = self.upsample_4(out4)
        out4 = self.conv4_1x1(out4)
        # print("4", out4.shape)

        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out


class AFIM(nn.Module):
    def __init__(self):
        super(AFIM, self).__init__()
        self.conv1 = nn.Conv2d(4, 128, kernel_size=3, padding=1,bias=False)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 4, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 1, kernel_size=1, padding=0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        # print(x)
        # 生成attention模块
        attention_weight = F.softmax(x, dim=1)
        featuremap = torch.mul(x, attention_weight)

        sum = torch.sum(featuremap, dim=1, keepdim=True)
        y = self.conv4(sum)
        y = self.conv5(y)
        y = self.conv6(y)
        y = y + sum
        return y


# J0---J4的attention层
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(4, 128, kernel_size=3, padding=1,bias=False)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 5, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        # print(x)
        # 生成attention模块
        attention_weight = F.softmax(x, dim=1)

        a0 = attention_weight[:, 0, :, :]
        a0 = a0.unsqueeze(dim=1)
        a1 = attention_weight[:, 1, :, :]
        a1 = a1.unsqueeze(dim=1)
        a2 = attention_weight[:, 2, :, :]
        a2 = a2.unsqueeze(dim=1)
        a3 = attention_weight[:, 3, :, :]
        a3 = a3.unsqueeze(dim=1)
        a4 = attention_weight[:, 4, :, :]
        a4 = a4.unsqueeze(dim=1)
        return a0, a1, a2, a3, a4

# 通过预测transmap和大气光得到的去雾图像
class J0(nn.Module):
    def __init__(self):
        super(J0, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1)

        self.pool = nn.MaxPool2d((5, 5), stride=2)
        self.fc = nn.Linear(65536, 1)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x, i):
        t = self.conv1(x)
        t = self.conv2(t)
        t = self.sigmoid(t)

        a = self.pool(x)
        a = x.view(x.size(0), -1)
        a = self.fc(a)
        eye = torch.ones_like(t)
        z = (eye-t)

        mid = torch.randn(1, 256, 256).to(opt.device)
        for x in range(a.size(0)):
            A = a[x][0]
            d = A * z[x]
            mid = torch.cat((mid, d), dim=0)
        mid = mid[1:, :, :]
        mid = mid.unsqueeze(1)
        mid = i - mid
        # print(mid.shape)
        j0 = self.sigmoid(torch.div(mid, t))
        # j0 = (i - a * (eye - t)) / t
        return j0


class J1(nn.Module):
    def __init__(self):
        super(J1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x, i):
        r = self.conv1(x)
        r = self.conv2(r)
        # print(r.shape)
        j1 = self.sigmoid1(torch.mul(r, i))
        return j1


class J2(nn.Module):
    def __init__(self):
        super(J2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x, i):
        r = self.conv1(x)
        r = self.conv2(r)
        # print(r.shape)
        j2 = self.sigmoid1(torch.add(r, i))
        return j2


class J3(nn.Module):
    def __init__(self):
        super(J3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x, i):
        r = self.conv1(x)
        r = self.conv2(r)
        # print(r.shape)
        j3 = self.sigmoid1(torch.pow(input=i, exponent=r))
        return j3


class J4(nn.Module):
    def __init__(self):
        super(J4, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x, i):
        r = self.conv1(x)
        r = self.conv2(r)
        # print(r.shape)
        j4 = torch.add(i, torch.mul(r, i)) * 2
        j4 = torch.clamp(j4, 1.0, 2.7)
        j4 = torch.log(j4)
        
        return j4


class DFMNet(nn.Module):
    def __init__(self):
        super(DFMNet, self).__init__()
        self.ResNext = ResNext()
        self.AFIM = AFIM()
        self.Attention = Attention()
        self.sigmoid1 = nn.Sigmoid()
        self.j0 = J0()
        self.j1 = J1()
        self.j2 = J2()
        self.j3 = J3()
        self.j4 = J4()

    def forward(self, x):
        MLF = self.ResNext(x)
        AMLIF0 = self.AFIM(MLF)
        AMLIF1 = self.AFIM(MLF)
        AMLIF2 = self.AFIM(MLF)
        AMLIF3= self.AFIM(MLF)
        AMLIF4 = self.AFIM(MLF)
        a0, a1, a2, a3, a4 = self.Attention(MLF)
        j0 = self.j0(AMLIF0, x)
        j1 = self.j1(AMLIF1, x)
        j2 = self.j2(AMLIF2, x)
        j3 = self.j3(AMLIF3, x)
        j4 = self.j4(AMLIF4, x)

        J0 = torch.mul(j0, a0)
        J1 = torch.mul(j1, a1)
        J2 = torch.mul(j2, a2)
        J3 = torch.mul(j3, a3)
        J4 = torch.mul(j4, a4)

        J = J0.add(J1).add(J2).add(J3).add(J4)
        # J = self.sigmoid1(J)
        return J,  j0, j1, j2, j3, j4


def main():
    x = torch.randn(3, 1, 256, 256)
    y = torch.randn(3, 3, 256, 256)
    net = DFMNet()
    # print(net)
    # out = net(x)
    j, j0, j1, j2, j3, j4 = net(y)
    print("out的形状：", j.shape)
    print(j0)


if __name__ == '__main__':
    main()

