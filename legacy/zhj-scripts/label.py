import os
from os import listdir, getcwd
from os.path import join
import random


#训练集+验证集比例 小于等于1
#训练集比例
double_train_property = 0.8
#验证集比例
double_val_property = 0.2

f=open('train.txt', 'w')
# 遍历数据路径dataset_dir
def traversal_dataset_dir(dateset_dir_txt):
    with open(dateset_dir_txt, 'r') as file:
        for path in file:
            path = path.rstrip('\n')
            if os.path.isdir(path):
                traversal_imgs(path)
            else:
                print("路径不存在：",path)
            
# 遍历路径下图片写入到train.txt
def traversal_imgs(inputdir):
    for dirpath, dirnames, filenames in os.walk(inputdir):
        for name in filenames:
            if '.jpg' in name:
                f.write(os.path.join(dirpath,name)+'\n')
                
        
traversal_dataset_dir('dataset_dir.txt')
f.close()
# 打开文件并读取所有行，然后计算行数
with open('train.txt', 'r') as file:
    lines = file.readlines()

# 打印行数
print(len(lines))
print("complete")
