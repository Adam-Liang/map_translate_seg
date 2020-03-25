import numpy as np
import shutil
import os
from PIL import Image
from data.image_folder import make_dataset
from src.util.my_util import get_inner_path

def labelpixels(img3D, label_list=[[239,238,236],[255,242,175],[170,218,255],[208,236,208],[255,255,255]]):

    # scale array
    s = 256**np.arange(img3D.shape[-1])

    # Reduce image and labels to 1D
    img1D = img3D.reshape(-1,img3D.shape[-1]).dot(s)
    label1D = np.dot(label_list, s)

    ret1D=np.zeros(img1D.shape)
    for i,num in enumerate(label1D):
        ret1D+=(img1D==num)*(i+1)
    ret2D=ret1D.reshape(img3D.shape[:-1])
    return ret2D
    # # # Use searchsorted to trace back label indices
    # # sidx = label1D.argsort()
    # return sidx[np.searchsorted(label1D, img1D, sorter=sidx)]

def make_seg_img_from_maps(maps_path,segs_path):
    maps=make_dataset(maps_path)
    if os.path.isdir(segs_path):
        segs=make_dataset(segs_path)
        if len(maps)==len(segs):
            return
        else:
            shutil.rmtree(segs_path)
    os.makedirs(segs_path)
    for map in maps:
        map_np=np.array(Image.open(map).convert("RGB"))
        seg_np=labelpixels(map_np).astype(np.uint8)
        seg_pil=Image.fromarray(seg_np)
        seg_path=os.path.join(segs_path,get_inner_path(map,maps_path))
        seg_pil.save(seg_path)

    pass

if __name__=="__main__":
    pass
    # maps_path="D:\\遥感项目\\数据集\\20191117第三批数据\\导出数据\\rs\\15"
    # segs_path="D:\\遥感项目\\数据集\\20191117第三批数据\\导出数据\\rs\\seg15"
    # make_seg_img_from_maps(maps_path, segs_path)