import torch
import numpy as np
from PIL import Image
import os
import os.path as osp
import cv2

def pred2gray(pred): # pred: tensor:bs*n_class*h*w
    bs,n_class,h,w=pred.size()
    pred=pred.data.cpu().numpy()
    gray = pred.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(bs, h, w)
    gray=torch.from_numpy(gray)
    return gray

def gray2rgb(gray,n_class=5,label_list=[[239,238,236],[255,242,175],[170,218,255],[208,236,208],[255,255,255]]): # gray: np:h*w
    h,w=gray.shape
    mask=[]
    rgb=np.zeros((h,w,3))
    for i in range(n_class):
        tmp=(gray==i)
        mask.append(tmp)
        rgb+=np.expand_dims(tmp,2).repeat(3,axis=2)*label_list[i]
    rgb=rgb.astype(np.uint8)
    return rgb


if __name__=='__main__':
    path="D:\\map_translate\\数据集\\20191117第三批数据\\sample_20191115\\taiwan_10_18_20191115_copy\\15_tiny\\test_seg\\15-7022-13708.png"
    gray=Image.open(path)
    gray=np.array(gray)
    rgb=gray2rgb(gray)
    rgb=Image.fromarray(rgb)
    rgb.save("D:\\map_translate\\数据集\\20191117第三批数据\\sample_20191115\\taiwan_10_18_20191115_copy\\15_tiny\\test_seg\\15-7022-13708___.png")