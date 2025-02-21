import os, time
import string
from pathlib import Path


def get_chat_file_name(input_str: str) -> str:
    """处理字符串：
    1) 剔除所有标点符号
    2) 截取前6个英文单词或中文字符
    3) 将空格替换为'-'
    
    Args:
        input_str (str): 输入字符串
        
    Returns:
        str: 处理后的字符串
    """
    
    # 1. 剔除标点符号（包括中文和英文）
    # 定义中文标点符号集合
    chinese_punctuation = '，。！？；：（）《》【】“”‘’、·…—'
    # 合并英文和中文标点符号
    all_punctuation = string.punctuation + chinese_punctuation
    # 创建翻译表并删除所有标点符号
    translator = str.maketrans('', '', all_punctuation)
    cleaned_str = input_str.translate(translator)
    
    # 2. 截取前6个单词/字符
    # 对于英文：按空格分割取前6个单词
    if any(char.isalpha() for char in cleaned_str):
        words = cleaned_str.split()[:5]
        truncated_str = ' '.join(words)
    # 对于中文：直接取前6个字符
    else:
        truncated_str = cleaned_str[:5]
    
    # 3. 将空格替换为'-'
    final_str = truncated_str.replace(' ', '-')
    
    return final_str


def get_timestamp():
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    return timestamp


def get_all_files_from_directory(directory_path, file_extension=None):
    """获取指定目录下的所有文件路径，包括子文件夹中的文件
    
    Args:
        directory_path (str): 目录路径
        file_extension (str, optional): 文件扩展名，如"md"。默认为None，表示获取所有文件
    
    Returns:
        list: 包含所有文件路径的列表
    """
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid directory path: {directory_path}")
    
    if file_extension:
        return [str(file) for file in path.rglob(f"*.{file_extension}") if file.is_file()]
    else:
        return [str(file) for file in path.rglob("*") if file.is_file()]

def print_data_sources(source_datas):
    full_content = ""
    print("\n\nsource_datas----------------------------------------------------------------source_datas")
    for n in source_datas:
        full_content += f"score: {n.score}\n\n{n.metadata}\n\n{n.text}\n----------------------------------------------------------------------------------------\n"
        print(f"{n.score}\n\n{n.metadata}\n\n{n.text}\n----------------------------------------------------------------------------------------\n")
    return full_content


if __name__ == "__main__":
    result = get_timestamp()
    print(result)