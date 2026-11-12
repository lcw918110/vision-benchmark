import os
from os import listdir, getcwd
from os.path import join
import random
import shutil
# 一定txt中指定类的txt
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
            # 精确匹配.txt后缀（不区分大小写）
            if name.lower().endswith('.txt'):
                txt_path = os.path.join(dirpath, name)
                
                try:
                    # 读取文件内容并过滤空行
                    with open(txt_path, 'r') as file:
                        valid_lines = [line.strip() for line in file if line.strip()]
                        
                        # 检查是否存在以0/1/2开头的行
                        has_target_line = len(valid_lines) > 0 and any(
                            #line.startswith(('0 ', '1 ', '2 '))
                            line.startswith(('0 '))
                        # 检查txt中是否全部以0开头的行
                        #has_target_line = len(valid_lines) > 0 and all(
                        #    line.startswith(('0 ')) 
                            for line in valid_lines
                        )
                        
                        # 当存在目标行时记录路径
                        if has_target_line:
                            f.write(os.path.join(dirpath, name) + '\n')
                            
                except Exception as e:
                    print(f"处理文件失败: {txt_path} | 错误: {str(e)}")
                    continue
                
        
traversal_dataset_dir('dataset_dir.txt')
f.close()

# 打开文件并读取所有行，然后计算行数
with open('train.txt', 'r') as file:
    lines = file.readlines()

# 打印行数
print(len(lines))
print("complete")
