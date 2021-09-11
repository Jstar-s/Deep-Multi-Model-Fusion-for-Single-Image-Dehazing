import torch, os, sys, torchvision, argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
from models import *
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from models.DM2Net import DFMNet

warnings.filterwarnings('ignore')
from option import opt, model_name, log_dir
from data_utils import *
from torchvision.models import vgg16

print('log_dir :', log_dir)
print('model_name:', model_name)

# 指定模型
models_ = {
    'DM2Net': DFMNet()
}
# 指定导入的数据集的DataLader对象，这些对象在 data_utils中
loaders_ = {
    'its_train': ITS_train_loader,
    'its_test': ITS_test_loader,
}
# 程序开始执行的时间
start_time = time.time()
# 指定训练的epoch数目
T = opt.steps


# 设置学习率的衰减公式


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


# 定义训练的函数实现
def train(net, loader_train, loader_test, optim, criterion):
    losses = []  # 定义保存loss的列表
    start_step = 0  # 初始化训练的迭代数
    max_ssim = 0  # 初始化最大ssim
    max_psnr = 0  # 初始化最大psnr
    ssims = []  # 定义保存ssim的列表
    psnrs = []  # 定义保存psnr的列表

    # opt.resume 的含义是是否是从0开始训练还是finetune，
    if opt.resume and os.path.exists(opt.model_dir):  # 模型所在的文件所在文件夹
        print(f'resume from {opt.model_dir}')  # 打印导入的模型
        ckp = torch.load(opt.model_dir)  # 导入模型
        losses = ckp['losses']  # 导出模型的loss
        net.load_state_dict(ckp['model'])  # 网络结构
        # 导入预训练的网络参数
        """"
        # 返回的是一个OrderDict，存储了网络结构的名字和对应的参数，
        paramters(需要反向传播的),_buffers,_modules和_state_dict_hooks
        """
        # w网络中预训练的各项参数
        start_step = ckp['step']  # 预训练的的轮数
        max_ssim = ckp['max_ssim']  # 预训练的max_ssim
        max_psnr = ckp['max_psnr']  # 预训练的max_psnr
        psnrs = ckp['psnrs']  # 预训练的所有psnr参数
        ssims = ckp['ssims']  # 预训练的所有ssims参数
        print(f'start_step:{start_step} start training ---')  # 打印开始寻连的轮数
    else:
        print('train from scratch *** ')  # 否证重初始状态开始训练
    for step in range(start_step + 1, opt.steps + 1):  # 从1开始或者从载入的模型开始训练  到训练的设置轮数 预设是100000
        net.train()  # 网络开始训练
        lr = opt.lr  # 指定学习率
        if not opt.no_lr_sche:  # 如果指定学习率衰减
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        # 动态调整优化器optimizer中的学习率
        haze, clear = next(iter(loader_train))  # 从数据集中取出要训练的图片和结果
        haze = haze.to(opt.device)
        clear = clear.to(opt.device)  # 把数据加载到GPU中加快训练速度
        out_j, out_j0, out_j1, out_j2, out_j3, out_j4 = net(haze)
        loss1 = criterion[0](out_j, clear)  # loss方法的第一个维度中使用的是L1loss使用的数据是out,
        loss2 = criterion[0](out_j0, clear)
        loss3 = criterion[0](out_j1, clear)
        loss4 = criterion[0](out_j2, clear)
        loss5 = criterion[0](out_j3, clear)
        loss6 = criterion[0](out_j4, clear)
        loss = 0.3*loss1 + 0.1*loss2 + 0.1*loss3 + 0.1*loss4 + 0.1*loss5 + 0.1*loss6

        optim.step()  # 参数更新
        optim.zero_grad()  # 梯度清零
        losses.append(loss.item())  # 保存每个step的loss
        print(
            f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)
        # 打印当前的loss , 15, 50/1000000   lr:0.001   time_used 单位是分钟
        with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
        	writer.add_scalar('data/loss',loss,step)
        # loss可视化代码
        # 如果达到了指定的轮数，使用测试集进行评价
        if step % opt.eval_step == 0:
            #  将网络设置为无反向传播模式， 使用test函数得到评价指标
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)
            # 打印当前的轮数和对应在测试集上的指标的值
            print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')
            # 可视化代码
            with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
            	writer.add_scalar('data/ssim',ssim_eval,step)
            	writer.add_scalar('data/psnr',psnr_eval,step)
            	writer.add_scalars('group',{
            		'ssim':ssim_eval,
            		'psnr':psnr_eval,
            		'loss':loss
            	},step)
            # 将所有评价指标的值保存在一个列表中
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            #       如果当前参数优于以往参数则保存模型数据
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                """
                模型中保存的数据有自定义的字段以及网络参数
                net.start_dict()
                """
                torch.save({
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict()
                }, opt.model_dir)

                # 在指定的文件夹中保存网络模型
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')
    # 保存模型的loss，等参数到numpy能读取的数据格式，方便再次读取和绘图
    np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)


#	测试部分代码
def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()  # 将网络设置成评估模式·
    torch.cuda.empty_cache()  # 优化代码清空部分gpu缓存
    ssims = []  # 初始化评价参数
    psnrs = []
    s=True
    #	对测试数据集进行读取
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device);
        targets = targets.to(opt.device)  # 将输入输出数据载入gpu
        pred, _1, _2, _3, _4, _5 = net(inputs)  # 网络得到的结果
        # # print(pred)
        # tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
        vutils.save_image(targets.cpu(), 'target.png')
        # vutils.save_image(pred.cpu(),'pred.png')		将测试结果保存
        ssim1 = ssim(pred, targets).item()  # 计算测试集的评价参数
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)  # 将测试集的评价参数保存
    psnrs.append(psnr1)
    # if (psnr1>max_psnr or ssim1 > max_ssim) and s :
    # 		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
    # 		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
    # 		s=False
    # # 保存数目前评价指标最好的测试结果的图片
    return np.mean(ssims), np.mean(psnrs)


#		返回测试集结果的平均值


if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = []  # 存放评价数据所需的loss的list
    criterion.append(nn.L1Loss().to(opt.device))  # 将训练过程中的L1loss加入其中
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    train(net, loader_train, loader_test, optimizer, criterion)





