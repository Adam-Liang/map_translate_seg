import os
import shutil
from tqdm import tqdm
from PIL import Image

'''拷贝并统一为png格式'''

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif',
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
def get_inner_path(file_path,floder_path):
    assert file_path[:len(floder_path)]==floder_path,"传入的文件不在文件夹中！[%s][%s]"%(file_path,floder_path)
    file_path=file_path[len(floder_path)+1:]
    return file_path
def make_floder_from_file_path(file_path):
    if not os.path.isdir(os.path.split(file_path)[0]):
        os.makedirs(os.path.split(file_path)[0])

if __name__=='__main__':
    # floder_list1=[r'D:\map_translate\数据集\CN10分析\use_repaint1',r'D:\map_translate\数据集\QH12分析\use_repaint1',
    #               r'D:\map_translate\数据集\HN14分析\使用数据-repaint1',r'D:\map_translate\数据集\WH16分析\二次筛选相关\二次筛选结果\使用的2700图',
    #               r'D:\map_translate\数据集\BJ18分析\use_repaint1']
    floder_list1=[r'/home/liangshuaizhe/dataset/CN10/use_repaint1',r'/home/liangshuaizhe/dataset/QH12/use_repaint1',
                  r'/home/liangshuaizhe/dataset/HN14/use_repaint1',r'/home/liangshuaizhe/dataset/WH16/use_repaint1',
                  r'/home/liangshuaizhe/dataset/BJ18/use_repaint1']
    target_list1=['CN10','QH12','HN14','WH16','BJ18']
    floder_list2=['trainA','trainB','train_seg','testA','testB','test_seg']

    # floder_new=r'D:\map_translate\数据集\AIDOMG-all\use_repaint1'
    floder_new=r'/home/liangshuaizhe/dataset/AIDOMG-all/use_repaint1'

    assert len(floder_list1)==len(target_list1)

    for f1 in floder_list1:

        for f2 in floder_list2:
            print(f'{f1},{f2},start')
            f_old=os.path.join(f1,f2)
            f_new=os.path.join(floder_new,f2,target_list1[floder_list1.index(f1)])

            imgs_old=make_dataset(f_old)
            for img_old in tqdm(imgs_old):
                img_new=os.path.join(f_new,get_inner_path(img_old,f_old))
                img_new=os.path.splitext(img_new)[0]+'.png'
                make_floder_from_file_path(img_new)
                if os.path.splitext(img_old)[1] != '.png':
                    _ = Image.open(img_old)
                    _.save(img_new)
                elif os.path.splitext(img_old)[1] == '.png':
                    shutil.copy2(img_old,img_new)


