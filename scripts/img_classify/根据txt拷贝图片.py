import os
import shutil
from tqdm import tqdm

def get_inner_path(file_path,floder_path):
    assert file_path[:len(floder_path)]==floder_path,"传入的文件不在文件夹中！[%s][%s]"%(file_path,floder_path)
    file_path=file_path[len(floder_path)+1:]
    return file_path
def make_floder_from_file_path(file_path):
    if not os.path.isdir(os.path.split(file_path)[0]):
        os.makedirs(os.path.split(file_path)[0])

if __name__=='__main__':
    train_ratio=1.0
    real_img_floder=r'D:\map_translate\数据集\HN14分析\使用数据-norepaint\A_all'
    # txt_paths = [r'D:\datasets\maps\统计各类图片数量\城市、田野、森林、河流、海洋\pick00.txt',r'D:\datasets\maps\统计各类图片数量\城市、田野、森林、河流、海洋\pick01.txt',r'D:\datasets\maps\统计各类图片数量\城市、田野、森林、河流、海洋\pick02.txt',r'D:\datasets\maps\统计各类图片数量\城市、田野、森林、河流、海洋\pick03.txt',r'D:\datasets\maps\统计各类图片数量\城市、田野、森林、河流、海洋\pick04.txt']
    txt_paths=[]
    for i in range(5):
        txt_paths.append(rf'D:\map_translate\数据集\HN14分析\使用数据-norepaint\城市、田野、森林、河流、海洋\pick0{i}.txt')
    txt_base_floder=r'D:\map_translate\数据集\HN14分析\使用数据-norepaint\floder_all'
    # new_floders=[r'D:\datasets\maps\统计各类图片数量\城市、田野、森林、河流、海洋\图片拷贝\0',r'D:\datasets\maps\统计各类图片数量\城市、田野、森林、河流、海洋\图片拷贝\1',r'D:\datasets\maps\统计各类图片数量\城市、田野、森林、河流、海洋\图片拷贝\2',r'D:\datasets\maps\统计各类图片数量\城市、田野、森林、河流、海洋\图片拷贝\3',r'D:\datasets\maps\统计各类图片数量\城市、田野、森林、河流、海洋\图片拷贝\4']
    new_floders=[]
    for i in range(5):
        new_floders.append(rf'D:\map_translate\数据集\HN14分析\使用数据-norepaint\城市、田野、森林、河流、海洋\图片拷贝\{i}')


    assert len(txt_paths)==len(new_floders)
    # for floder in new_floders:
    #     if not os.path.isdir(floder):
    #         os.makedirs(floder)
    if not real_img_floder:
        pass
        # for i in range(len(txt_paths)):
        #     img_list=[]
        #     for row in open(txt_paths[i], 'r'):
        #         img_list.append(row[:len(row) - 1])
        #     print(len(img_list))
        #     for img in tqdm(img_list):
        #         shutil.copy(img,new_floders[i])
    else:
        for i in range(len(txt_paths)):
            img_list=[]
            for row in open(txt_paths[i], 'r'):
                img_list.append(row[:len(row) - 1])
                img_list[-1]=os.path.join(real_img_floder,get_inner_path(img_list[-1],txt_base_floder))
                img_list[-1]=os.path.splitext(img_list[-1])[0]+'.jpg'

            print(len(img_list))
            for j in tqdm(range(len(img_list))):
                img = img_list[j]
                if j<(len(img_list)*train_ratio):
                    img_new=get_inner_path(img,real_img_floder)
                    img_new=os.path.join(new_floders[i]+'_train',img_new)

                    img_new = os.path.split(img_new)[0]+'-'+os.path.split(img_new)[1]
                    make_floder_from_file_path(img_new)
                    shutil.copy(img,img_new)
                else:
                    img_new = get_inner_path(img, real_img_floder)
                    img_new = os.path.join(new_floders[i] + '_test', img_new)
                    make_floder_from_file_path(img_new)
                    shutil.copy(img, img_new)