'''p2pHD，增加参数量以使其和联合方法参数量差不多'''
'''需增至约1.28倍，直接改变args.ngf'''
import os
from os import path as osp
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
# sys.path.append(osp.join(sys.path[0], '../../../'))
import time
import torch
import torch.nn as nn
from src.utils.train_utils import model_accelerate, get_device, mean, get_lr
from src.pix2pixHD.train_config import config
from src.pix2pixHD.networks import get_G, get_D, get_E
from torch.optim import Adam
from src.pix2pixHD.hinge_lr_scheduler import get_hinge_scheduler
from src.utils.logger import ModelSaver, Logger
from src.datasets import get_pix2pix_maps_dataloader
from src.pix2pixHD.utils import get_edges, label_to_one_hot, get_encode_features
from src.utils.visualizer import Visualizer
from tqdm import tqdm
from torchvision import transforms
from src.pix2pixHD.criterion import get_GANLoss, get_VGGLoss, get_DFLoss, get_low_level_loss
from tensorboardX import SummaryWriter
from src.pix2pixHD.utils import from_std_tensor_save_image, create_dir
from pix2pixHD.eval_img2map import eval, get_fid
from src.data.image_folder import make_dataset
import shutil
import numpy as np
from PIL import Image

# from src.pix2pixHD.deeplabv3.model.deeplabv3 import DeepLabV3
from src.pix2pixHD.deeplabv3plus.deeplabv3plus import Configuration
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.deeplabv3plus_seg import deeplabv3plus
import torch.nn.functional as F




def train(args, get_dataloader_func=get_pix2pix_maps_dataloader):
    logger = Logger(save_path=args.save, json_name='img2map')
    epoch_now = len(logger.get_data('G_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D', 'E', 'G_optimizer', 'D_optimizer', 'E_optimizer',
                                        'G_scheduler', 'D_scheduler', 'E_scheduler','DLV3P',"DLV3P_global_optimizer",
                                        "DLV3P_backbone_optimizer","DLV3P_global_scheduler","DLV3P_backbone_scheduler"])
    visualizer = Visualizer(keys=['image', 'encode_feature', 'fake', 'label', 'instance'])
    sw = SummaryWriter(args.tensorboard_path)

    args.ngf=int((1.28**0.5)*args.ngf)
    args.ndf = int((1.28 ** 0.5) * args.ndf)
    G = get_G(args)
    D = get_D(args)
    model_saver.load('G', G)
    model_saver.load('D', D)

    print(f"{args.ngf}")
    params_G=sum([param.nelement() for param in G.parameters()])
    params_D = sum([param.nelement() for param in D.parameters()])
    print(f"{params_G}, {params_D}")
    print(f"{params_G}")
    # sys.exit(0) # 测完退出

    G_optimizer = Adam(G.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D_optimizer = Adam(D.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))

    model_saver.load('G_optimizer', G_optimizer)
    model_saver.load('D_optimizer', D_optimizer)

    G_scheduler = get_hinge_scheduler(args, G_optimizer)
    D_scheduler = get_hinge_scheduler(args, D_optimizer)

    model_saver.load('G_scheduler', G_scheduler)
    model_saver.load('D_scheduler', D_scheduler)

    device = get_device(args)

    GANLoss = get_GANLoss(args)

    if args.use_ganFeat_loss:
        DFLoss = get_DFLoss(args)
    if args.use_vgg_loss:
        VGGLoss = get_VGGLoss(args)
    if args.use_low_level_loss:
        LLLoss = get_low_level_loss(args)


    for epoch in range(epoch_now, args.epochs):
        G_loss_list = []
        D_loss_list = []

        data_loader = get_dataloader_func(args, train=True)
        data_loader = tqdm(data_loader)

        for step, sample in enumerate(data_loader):
            # 然后训练GAN
            imgs=sample['A'].to(device)
            maps = sample['B'].to(device)

            # train the Discriminator
            D_optimizer.zero_grad()
            reals_maps = torch.cat([imgs.float(), maps.float()], dim=1)

            fakes = G(imgs).detach()
            fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)

            D_real_outs = D(reals_maps)
            D_real_loss = GANLoss(D_real_outs, True)

            D_fake_outs = D(fakes_maps)
            D_fake_loss = GANLoss(D_fake_outs, False)

            D_loss = 0.5 * (D_real_loss + D_fake_loss)
            D_loss = D_loss.mean()
            D_loss.backward()
            D_loss = D_loss.item()
            D_optimizer.step()

            # train generator and encoder
            G_optimizer.zero_grad()
            # fakes = G(imgs,input_2,input_3)
            fakes = G(imgs)
            fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)
            D_fake_outs = D(fakes_maps)

            gan_loss = GANLoss(D_fake_outs, True)

            G_loss = 0
            G_loss += gan_loss
            gan_loss = gan_loss.mean().item()

            if args.use_vgg_loss:
                vgg_loss = VGGLoss(fakes, imgs)
                G_loss += args.lambda_feat * vgg_loss
                vgg_loss = vgg_loss.mean().item()
            else:
                vgg_loss = 0.

            if args.use_ganFeat_loss:
                df_loss = DFLoss(D_fake_outs, D_real_outs)
                G_loss += args.lambda_feat * df_loss
                df_loss = df_loss.mean().item()
            else:
                df_loss = 0.

            if args.use_low_level_loss:
                ll_loss = LLLoss(fakes, maps)
                G_loss += args.lambda_feat * ll_loss
                ll_loss = ll_loss.mean().item()
            else:
                ll_loss = 0.

            G_loss = G_loss.mean()
            G_loss.backward()
            G_loss = G_loss.item()

            G_optimizer.step()

            data_loader.write(f'Epochs:{epoch} | Dloss:{D_loss:.6f} | Gloss:{G_loss:.6f}'
                              f'| GANloss:{gan_loss:.6f} | VGGloss:{vgg_loss:.6f} | DFloss:{df_loss:.6f} '
                              f'| LLloss:{ll_loss:.6f} | lr:{get_lr(G_optimizer):.8f}')

            G_loss_list.append(G_loss)
            D_loss_list.append(D_loss)

            # display
            if args.display and step % args.display == 0:
                visualizer.display(transforms.ToPILImage()(imgs[0].cpu()), 'image')
                visualizer.display(transforms.ToPILImage()(fakes[0].cpu()), 'fake')
                visualizer.display(transforms.ToPILImage()(maps[0].cpu()), 'label')

            # tensorboard log
            if args.tensorboard_log and step % args.tensorboard_log == 0:
                total_steps = epoch * len(data_loader) + step
                sw.add_scalar('Loss/G', G_loss, total_steps)
                sw.add_scalar('Loss/D', D_loss, total_steps)
                sw.add_scalar('Loss/gan', gan_loss, total_steps)
                sw.add_scalar('Loss/vgg', vgg_loss, total_steps)
                sw.add_scalar('Loss/df', df_loss, total_steps)
                sw.add_scalar('Loss/ll', ll_loss, total_steps)


                sw.add_scalar('LR/G', get_lr(G_optimizer), total_steps)
                sw.add_scalar('LR/D', get_lr(D_optimizer), total_steps)


                from util.util import tensor2im
                sw.add_image('img2/realA', tensor2im(imgs.data), step,dataformats='HWC')
                sw.add_image('img2/fakeB', tensor2im(fakes.data), step,dataformats='HWC')
                sw.add_image('img2/realB', tensor2im(maps.data), step,dataformats='HWC')


        D_scheduler.step(epoch)
        G_scheduler.step(epoch)

        if epoch % 10 == 0 or epoch == (args.epochs-1):
            fid, iou = eval(args, model=G, data_loader=get_dataloader_func(args, train=False))
            logger.log(key='FID', data=fid)
            if fid > logger.get_max(key='FID'):
                model_saver.save(f'G_{fid:.4f}', G)
                model_saver.save(f'D_{fid:.4f}', D)
            sw.add_scalar('fid/fid', fid, epoch)

        logger.log(key='D_loss', data=sum(D_loss_list) / float(len(D_loss_list)))
        logger.log(key='G_loss', data=sum(G_loss_list) / float(len(G_loss_list)))
        logger.save_log()

        model_saver.save('G', G)
        model_saver.save('D', D)

        model_saver.save('G_optimizer', G_optimizer)
        model_saver.save('D_optimizer', D_optimizer)

        model_saver.save('G_scheduler', G_scheduler)
        model_saver.save('D_scheduler', D_scheduler)

    # （若载入训练完的文件夹,以上训练过程不会被执行，会跳至这一步；若正常训练至这一步，该步不会被执行）
    if epoch_now == args.epochs:
        print('Got final models!')
        fid = eval(args, model=G, data_loader=get_dataloader_func(args, train=False))
        # logger.log(key='FID', data=fid)
        # if fid > logger.get_max(key='FID'):
        #     model_saver.save(f'G_{fid:.4f}', G)
        #     model_saver.save(f'D_{fid:.4f}', D)
        # sw.add_scalar('fid/fid', fid, epoch)
        print('Test finish!')


if __name__ == '__main__':
    args = config()
    assert args.feat_num == 0
    assert args.use_instance == 0

    from src.pix2pixHD.myutils import seed_torch

    print(f'set seed as {args.seed}!')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
