import os
from PIL import Image

old=r'D:\map_translate\写作相关\latex内容\latex内容\配图\素材\131967335_tostand_new\A.tif'

assert os.path.splitext(old)[1]!='.png'
img=Image.open(old)
img.save(os.path.splitext(old)[0]+'.png')