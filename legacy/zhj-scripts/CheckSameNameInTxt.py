import os
from collections import defaultdict

# 输入 train.txt 文件路径
train_file_path = 'train.txt'

# 读取 train.txt 中的图片路径
with open(train_file_path, 'r') as f:
    image_paths = f.readlines()

# 清除路径两边的空格和换行符
image_paths = [path.strip() for path in image_paths]

# 使用 defaultdict 存储文件名及其对应的路径
file_dict = defaultdict(list)

# 遍历每个图片路径，将其文件名作为键，路径作为值
for image_path in image_paths:
    # 获取文件名
    file_name = os.path.basename(image_path)
    # 将文件路径添加到对应文件名的列表中
    file_dict[file_name].append(image_path)

# 输出同名图片和路径
same_name_count = 0
for file_name, paths in file_dict.items():
    if len(paths) > 1:  # 如果某个文件名对应多个路径，说明有同名图片
        same_name_count += 1
        print(f"File name: {file_name}")
        for path in paths:
            print(f"  Path: {path}")

# 输出共有多少同名图片
print(f"\nTotal number of different same-name images: {same_name_count}")
