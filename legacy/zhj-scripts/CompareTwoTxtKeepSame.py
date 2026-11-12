import os
# 对比两个txt保留相同的
def read_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def compare_and_keep_entries(file1, file2):
    # 读取文件
    files1 = read_file(file1)
    files2 = set(os.path.basename(line) for line in read_file(file2))
    
    # 保留同名文件路径
    files1_filtered = [line for line in files1 if os.path.basename(line) in files2]
    
    # 写回file1，覆盖原内容
    with open(file1, 'w') as f:
        for line in files1_filtered:
            f.write(line + '\n')
    
    # 返回被保留的文件名集合
    kept_files = set(os.path.basename(line) for line in files1_filtered)
    return kept_files

# 示例使用
file1_path = './batch_8_in_testdataset.txt'
file2_path = './batch_violation_old.txt'

kept_files = compare_and_keep_entries(file1_path, file2_path)

# 输出被保留的文件名和数量
print("被保留的文件:", kept_files)
print(f"被保留的文件数量: {len(kept_files)}")