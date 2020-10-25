import os
from PIL import Image
from tqdm import tqdm
import numpy as np

# def rgb2gray(array): # 这函数用起来有问题，为什么呢？
#     assert len(array.shape)==3 and array.shape[2]==3
#     array=array[:,:,0]*0.299 + array[:,:,1]*0.587 + array[:,:,2]*0.114
#     return array

if __name__ == '__main__':

    img_old = r'D:\map_translate\写作相关\latex内容\latex内容\配图\素材\5359526944-WH\5359526944-A.tif'
    mask_old = r'D:\map_translate\写作相关\latex内容\latex内容\配图\素材\5359526944-WH\5359526944-seg.png'
    dir_new = os.path.splitext(img_old)[0]+'_to_mutigray'
    num=10

    mask=Image.open(mask_old)
    mask=np.array(mask)
    pic=Image.open(img_old)
    pic=pic.convert('1')
    pic=np.array(pic)

    pic=Image.fromarray(pic,mode='L')
    pic.save(os.path.join(dir_new, str(111) + '.jpg'))

    for i in range(num):
        pic_tmp=np.copy(pic)
        a=[]
        for j in range(3):
            a.append(np.full_like(mask,np.random.randint(-128,128)) )
            pic_tmp+=a[-1]* (mask==j)
        # pic_tmp= pic_tmp.astype(np.int32)+np.random.randint(-8,8,pic_tmp.shape)
        # pic_tmp[pic_tmp<0]=0
        # pic_tmp[pic_tmp > 255] = 255
        pic_tmp.astype(np.uint8)
        pic_tmp=Image.fromarray(pic_tmp,mode='L')
        # pic_tmp.mode='L'
        if not os.path.isdir(dir_new):
            os.makedirs(dir_new)
        pic_tmp.save(os.path.join(dir_new,str(i)+'.jpg'))





    # dir_old = r'D:\map_translate\写作相关\latex内容\参考代码\about_rs\Building-Extraction-master\Building-Extraction-master\data\vaihingen-data-batch\validation\image'
    #
    # imgs_old = make_dataset(dir_old)
    # for img in tqdm(imgs_old):
    #     assert os.path.splitext(img)[1] != '.png'
    #     img_new = Image.open(img)
    #     img_new.save(os.path.splitext(img)[0] + '.png')