import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
crop_size= 256

#   图片展示函数
def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''

        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()


#   数据集加载函数
class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,train,size,format='.png'):
        super(RESIDE_Dataset,self).__init__()
        self.size=size
        # print('crop size',size)
        self.train=train
        self.format=format      # 文件后缀名
        self.haze_imgs_dir=os.listdir(os.path.join(path,'hazy'))     # 将加载路径和haze进行拼接找到指定文件夹
        self.haze_imgs=[os.path.join(path,'hazy',img) for img in self.haze_imgs_dir]
        # for im in self.haze_imgs_dir:
        #     print(im)
        # 拼接图片的完整路径
        self.clear_dir=os.path.join(path, 'clear')          # 拼接清晰图片所在文件夹的完整路径
        self.trans = os.path.join(path, 'trans')

    def __getitem__(self, index):
        haze_image = self.haze_imgs[index]
        # print("haze_image", haze_image)
        haze=Image.open(self.haze_imgs[index])      #    根据索引加载指定的图片
        if isinstance(self.size,int):       #   对图片进行过滤，剔除损坏的文件
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,200)
                haze=Image.open(self.haze_imgs[index])
                #       如果图片的尺寸小于指定尺寸则另外随机打开一张图片作为代替
        img=self.haze_imgs[index] #       将图片的完整路径和名字读取出来
        id=img.split('/')[-1].split('_')[0]         #    对图片名进行裁切，只取最后的文件编号
        # 在linux下要修改为\
        clear_name=id+self.format           #   找到对应的清晰图片的文件名
        # print("clear_name", clear_name)
        clear=Image.open(os.path.join(self.clear_dir,clear_name))   # 将文件名进行拼接
        clear=tfs.CenterCrop(haze.size[::-1])(clear)        # 对图片进行中心裁剪
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
            # trans=FF.crop(trans, i,j,h,w)
        haze,clear, =self.augData(haze.convert("RGB"), clear.convert("RGB"))
        # print(clear.shape)

        return haze, clear,


    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        # data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        # target = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(target)
        return data, target

    def __len__(self):
        return len(self.haze_imgs)


class RESIDE_Dataset_test(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_Dataset_test,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format      # 文件后缀名
        self.haze_imgs_dir=os.listdir(os.path.join(path,'hazy'))        # 将加载路径和haze进行拼接找到指定文件夹
        self.haze_imgs=[os.path.join(path,'hazy',img) for img in self.haze_imgs_dir]
        # 拼接图片的完整路径
        self.clear_dir=os.path.join(path,'clear')          # 拼接清晰图片所在文件夹的完整路径

    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])      #    根据索引加载指定的图片
        if isinstance(self.size,int):       #   对图片进行过滤，剔除损坏的文件
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,20000)
                haze=Image.open(self.haze_imgs[index])
                #       如果图片的尺寸小于指定尺寸则另外随机打开一张图片作为代替
        img=self.haze_imgs[index]       #       将图片的完整路径和名字读取出来
        id=img.split('/')[-1].split('_')[0]         #    对图片名进行裁切，只取最后的文件编号
        clear_name=id+self.format           #   找到对应的清晰图片的文件名
        clear=Image.open(os.path.join(self.clear_dir,clear_name))   # 将文件名进行拼接
        clear=tfs.CenterCrop(haze.size[::-1])(clear)        # 对图片进行中心裁剪
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear

    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        # data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target

    def __len__(self):
        return len(self.haze_imgs)


import os
pwd=os.getcwd()
print(pwd)
# path = 'E:\\pycode\\Jiahao\\F2MNet\\data'

path='/home/yk/zjh_file/project_art/CSBNet/data'#path to your 'data' folder
ITS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/ITS',train=True,size=256),batch_size=2, shuffle=True)
ITS_test_loader=DataLoader(dataset=RESIDE_Dataset_test(path+'/RESIDE/SOTS/indoor',train=False,size=256), batch_size=2, shuffle=False)


def main():
    haze, clear = next(iter(ITS_train_loader))
    tensorShow(haze, "ab")
    tensorShow(clear, "cd")
    test, gt = next(iter(ITS_test_loader))
    tensorShow(test, "xy")
    tensorShow(gt, "qw")


if __name__ == "__main__":
    main()
