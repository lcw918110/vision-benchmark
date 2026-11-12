import os
import glob
import sys
#统计dataset_dir.txt文档中的路径下所有的图片以及标签个数
def find_txt_files(directory):
    # 使用glob模块查找所有的.txt文件
    txt_files = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)
    return txt_files


def get_txt_lines(file_path):
    obj_name_list = []
    with open(file_path, 'r') as file:
        # 逐行读取文件内容
        for line in file:
            if line.split(" ")[0] == "\n":
                pass
            else:
                obj_name_list.append(line.split(" ")[0])
           
    return obj_name_list


def get_obj_names(file_path):
    obj_name_list = []
    # 打开文件，使用 'r' 表示读取模式
    with open(file_path, 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 输出每一行的内容
            obj_name_list.append(line.split("\n")[0])
    return obj_name_list


obj_data_path = "obj.names"
directory_path = "dataset_dir.txt"
# obj_data_path = "obj.names"
# directory_path = "../riding_violation"

obj_data_list = get_obj_names(obj_data_path)
variables = {}
pictures = {}
file_num = 0

# 循环创建变量
print(obj_data_list)
for i in range(obj_data_list.__len__()):
    key = obj_data_list[i]
    value = 0
    variables[key] = value
    pictures[key] = value
    
with open(directory_path, 'r') as file:
    for line in file:
    	  directory = line.strip()
    	  txt_files = find_txt_files(directory)
    	  #print(txt_files)
    	  for txt_files_file in txt_files:
    	  	  file_num += 1
    	  	  obj_list = get_txt_lines(txt_files_file)
    	  	  for _ in range(obj_data_list.__len__()):
    	  	  	  flag = True
    	  	  	  for __ in obj_list:
    	  	  	  	  if int(__) == _:
    	  	  	  	  	  if flag:
    	  	  	  	  	  	  pictures[obj_data_list[_]] += 1
    	  	  	  	  	  flag = False
    	  	  	  	  	  variables[obj_data_list[_]] += 1

print("已标注标签个数：{}".format(variables))
print("含目标图像总数：{}".format(pictures))
print("图像总数：{}".format(file_num))