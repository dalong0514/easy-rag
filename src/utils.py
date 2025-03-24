import os, time
import string
from pathlib import Path

def utils_remove_punctuation(input_str: str) -> str:
    """剔除字符串中的所有标点符号（包括中文和英文）
    
    删除输入字符串中的所有英文和中文标点符号，返回纯文本。
    
    Args:
        input_str (str): 要处理的输入字符串
        
    Returns:
        str: 移除所有标点符号后的字符串
    """
    # 定义中文标点符号集合
    chinese_punctuation = '，。`！？；：（）《》【】""''、·…—'
    # 合并英文和中文标点符号
    all_punctuation = string.punctuation + chinese_punctuation
    # 创建翻译表并删除所有标点符号
    translator = str.maketrans('', '', all_punctuation)
    return input_str.translate(translator)

def get_chat_file_name(input_str: str) -> str:
    """处理字符串生成规范的文件名
    
    将输入字符串处理为适合作为文件名的格式：
    1) 剔除所有标点符号
    2) 截取前6个英文单词或中文字符
    3) 将空格替换为'-'
    
    Args:
        input_str (str): 输入字符串
        
    Returns:
        str: 处理后适合作为文件名的字符串
    """
    
    cleaned_str = utils_remove_punctuation(input_str)
    
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
    """获取当前时间戳
    
    返回格式化的当前时间戳，格式为'YYYYMMDDHHmmSS'。
    
    Returns:
        str: 格式化的时间戳字符串
    """
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    return timestamp


def get_all_files_from_directory(directory_path, file_extension=None):
    """获取指定目录下的所有文件路径，包括子文件夹中的文件
    
    递归搜索指定目录及其子目录，返回符合扩展名条件的所有文件路径。
    
    Args:
        directory_path (str): 目录路径
        file_extension (str, optional): 文件扩展名，如"md"。默认为None，表示获取所有文件
    
    Returns:
        list: 包含所有符合条件的文件路径的列表
        
    Raises:
        ValueError: 如果提供的目录路径无效
    """
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid directory path: {directory_path}")
    
    if file_extension:
        return [str(file) for file in path.rglob(f"*.{file_extension}") if file.is_file()]
    else:
        return [str(file) for file in path.rglob("*") if file.is_file()]

def print_data_sources(source_datas):
    """打印和格式化检索结果数据
    
    将检索到的源数据格式化输出，同时返回格式化的完整内容字符串。
    
    Args:
        source_datas (list): 源数据节点列表，通常是检索返回的结果
        
    Returns:
        str: 格式化后的完整内容字符串
    """
    full_content = ""
    print("\n\nsource_datas----------------------------------------------------------------source_datas")
    for i, n in enumerate(source_datas):
        full_content += f"score: {n.score}\n\n{n.metadata}\n\n{{[indexpage {i} begin]{n.text}[indexpage {i} end]}}\n----------------------------------------------------------------------------------------\n"
        print(f"{n.score}\n\n{n.metadata}\n\nf\"{{[indexpage {i} begin]{n.text}[indexpage {i} end]}}\"\n----------------------------------------------------------------------------------------\n")
    return full_content


if __name__ == "__main__":
    result = get_timestamp()
    print(result)