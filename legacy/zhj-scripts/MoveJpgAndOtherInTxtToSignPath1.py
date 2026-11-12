import os
import shutil

def copy_files_from_txt(txt_file_path, dest_folder):
    """
    从txt文件中读取图片路径，并将这些图片以及和图片同名的文件复制到指定文件夹（假设这些文件在与图片相同的目录中）
    :param txt_file_path: 包含文件路径的txt文件路径
    :param dest_folder: 目标文件夹路径
    """
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 打开txt文件并逐行读取
    with open(txt_file_path, 'r') as file:
        for line in file:
            # 去除每行末尾的换行符
            file_path = line.strip()
            # 检查文件是否存在
            if os.path.isfile(file_path):
                # 获取文件名和扩展名
                file_dir, file_name_with_ext = os.path.split(file_path)
                file_name, file_ext = os.path.splitext(file_name_with_ext)
                
                # 构造目标文件路径
                dest_file_path = os.path.join(dest_folder, file_name_with_ext)
                # 复制文件
                shutil.move(file_path, dest_file_path)
                # print(f"Copied {file_path} to {dest_file_path}")
                
                # 查找并复制与图片同名的其他文件
                #for root, dirs, files in os.walk(file_dir):
                    #for f in files:
                        # 检查文件是否以相同的文件名开始，但不包括当前文件本身，并且扩展名不同
                        #f_name, f_ext = os.path.splitext(f)
                        #if f_name == file_name and f != file_name_with_ext:
                            #src_related_file_path = os.path.join(root, f)
                            
                            # 构造图片外目标文件路径
                            #dest_related_file_path = os.path.join(dest_folder, f)
                            #print(src_related_file_path)
                             # 构造目标文件的完整路径（保持在相同的子目录结构中，如果有的话）
                             # 如果不需要保持子目录结构，可以直接使用 dest_folder 和 f
                            #rel_path = os.path.relpath(root, file_dir)
                            # 复制相关文件
                            #shutil.copy(src_related_file_path, dest_related_file_path)
            else:
                print(f"File {file_path} not found.")

# 使用示例
txt_file_path = './batch_8_in_testdataset.txt'  # 图片路径
dest_folder = './batch_8_in_testdataset_same'  # 目标文件夹
copy_files_from_txt(txt_file_path, dest_folder)