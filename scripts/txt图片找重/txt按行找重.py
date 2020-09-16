import os
import shutil
from tqdm import tqdm

def find_same_element(set1,set2):
    ret=[]
    for ele in set1:
        if ele in set2:
            ret.append(ele)
    return ret


if __name__=='__main__':
    txt_paths = []
    for i in range(5):
        txt_paths.append(fr"D:\map_translate\数据集\HN14分析\使用数据-norepaint\城市、田野、森林、河流、海洋\pick0{i}.txt")

    txt_sets=[]
    for i in range(len(txt_paths)):
        txt_sets.append(set())
        for row in open(txt_paths[i], 'r'):
            txt_sets[i].add(row)
        print(len(txt_sets[i]))

    for i in range(len(txt_paths)):
        for j in range(i+1,len(txt_paths)):
            eles=find_same_element(txt_sets[i],txt_sets[j])
            if eles:
                print(f"第{i+1}个集合与第{j+1}个集合之间有{len(eles)}个重复")
                for ele in eles:
                    print(ele)