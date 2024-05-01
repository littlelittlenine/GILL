import json
import os
from tqdm import tqdm

def process_json_files(directory, output_file):
    # 检查输出文件是否已存在，如果存在则删除
    # Check if the output file exists, and delete it if it does
    if os.path.exists(output_file):
        os.remove(output_file)

    # 创建并写入列标题
    # Create and write the column headers
    with open(output_file, 'a') as out_file:
        out_file.write('caption\timage\n')

    # 遍历指定目录下的所有文件
    # Iterate over all files in the specified directory
    # os.listdir() 函数用于获取指定目录下的所有文件和子目录的名称列
    for filename in tqdm(os.listdir(directory),desc="Parsing dataset..."):
        if filename.endswith('.json'):
            # 构建完整的文件路径
            # Construct the full file path
            filepath = os.path.join(directory, filename)

            # 打开并读取 JSON 文件
            # Open and read the JSON file
            with open(filepath, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    continue

                # 检查 status 字段是否为 'success'
                # Check if the status field is 'success'
                if data.get('status') == 'success':
                    # 提取 caption 和 key 字段的值
                    # Extract the values of the caption and key fields
                    caption = data.get('caption', '')
                    key = data.get('key', '') + '.jpg'

                    # 将提取的数据写入到输出文件
                    # Write the extracted data to the output file
                    with open(output_file, 'a') as out_file:
                        out_file.write(f"{caption}\t{key}\n")


# 调用函数，指定目录和输出文件的路径
# Call the function, specifying the directory and output file paths
process_json_files('/data/little-nine/GILL/datasets/cc3m/validation', '/data/little-nine/GILL/datasets/cc3m_val.tsv')