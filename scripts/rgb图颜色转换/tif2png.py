import os
from PIL import Image
from rgb2gray import make_dataset
from tqdm import tqdm

if __name__=='__main__':
    

    dir_old=r'D:\map_translate\写作相关\latex内容\参考代码\about_rs\Building-Extraction-master\Building-Extraction-master\data\vaihingen-data-batch\validation\image'

    imgs_old = make_dataset(dir_old)
    for img in tqdm(imgs_old):
        assert os.path.splitext(img)[1]!='.png'
        img_new=Image.open(img)
        img_new.save(os.path.splitext(img)[0]+'.png')