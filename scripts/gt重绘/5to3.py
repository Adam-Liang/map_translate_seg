import shutil
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np




IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def get_inner_path(file_path, floder_path):
    assert file_path[:len(floder_path)] == floder_path, "传入的文件不在文件夹中！[%s][%s]" % (file_path, floder_path)
    file_path = file_path[len(floder_path) + 1:]
    return file_path

def make_floder_from_file_path(file_path):
    if not os.path.isdir(os.path.split(file_path)[0]):
        os.makedirs(os.path.split(file_path)[0])




def make_rgb_gt(dir_seg, dir_new): # seg 文件为256*256灰度label图
    if not os.path.isdir(dir_new):
        os.makedirs(dir_new)
        print("Flod data floder creating!")
    num = 0

    imgs_old = make_dataset(dir_seg)
    for img in tqdm(imgs_old):
        img_inner = get_inner_path(img, dir_seg)
        old = Image.open(img)
        old=np.array(old)
        new=np.array(old)
        new[new==3]=0
        new[new==4]=1
        new[new == 5] = 0
        if new.max()>2:
            print(f'new.max:{new.max()}')
        new=Image.fromarray(new)
        make_floder_from_file_path(os.path.join(dir_new, img_inner))
        new.save(os.path.join(dir_new, img_inner))
        num += 1
    print("New data floder created! %d img was processed" % num)


if __name__ == "__main__":
    flag = 1
    # 首先解析文件路径
    path_seg = r"D:\map_translate\数据集\CN10分析\seg_5label"
    # path_new = r"D:\map_translate\看看效果\0426TW16_1708图_celvs,epoch200\real_seg_new"
    # path_new = path_seg+'_repaint'
    path_new = path_seg+'_to3'

    make_rgb_gt(path_seg, path_new)
    print("finish!")
