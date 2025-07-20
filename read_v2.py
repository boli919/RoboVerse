import pickle
import numpy as np
import os
from datetime import datetime

def read_pkl_file(file_path, output_file=None):
    """
    读取pkl文件并将键名和键值保存到txt文件
    
    Args:
        file_path (str): pkl文件的路径
        output_file (str): 输出txt文件的路径，如果为None则自动生成
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            error_msg = f"错误: 文件 {file_path} 不存在"
            print(error_msg)
            return
        
        # 生成输出文件名
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{base_name}_analysis_{timestamp}.txt"
        
        # 读取pkl文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 打开输出文件
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(f"PKL文件分析报告\n")
            out_f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            out_f.write(f"源文件: {file_path}\n")
            out_f.write("=" * 60 + "\n\n")
            
            out_f.write(f"成功读取文件: {file_path}\n")
            out_f.write("=" * 60 + "\n")
            
            # 检查数据类型
            out_f.write(f"数据类型: {type(data)}\n")
            out_f.write("=" * 60 + "\n")
        
            # 如果是字典类型，打印所有键值对
            if isinstance(data, dict):
                out_f.write("字典中的键名和对应的键值:\n")
                out_f.write("-" * 40 + "\n")
                
                for key, value in data.items():
                    out_f.write(f"键名: {key}\n")
                    out_f.write(f"键值类型: {type(value)}\n")
                    
                    # 根据值的类型进行不同的处理
                    if isinstance(value, (str, int, float, bool)):
                        out_f.write(f"键值: {value}\n")
                    elif isinstance(value, (list, tuple)):
                        out_f.write(f"键值长度: {len(value)}\n")
                        if len(value) > 0:
                            out_f.write(f"第一个元素: {value[0]} (类型: {type(value[0])})\n")
                            if len(value) > 5:
                                out_f.write(f"前5个元素: {value[:5]}\n")
                            else:
                                out_f.write(f"所有元素: {value}\n")
                    elif isinstance(value, np.ndarray):
                        out_f.write(f"NumPy数组形状: {value.shape}\n")
                        out_f.write(f"数据类型: {value.dtype}\n")
                        if value.size > 0:
                            out_f.write(f"数组内容预览: {value.flatten()[:10]}...\n")  # 显示前10个元素
                    else:
                        out_f.write(f"键值: {str(value)[:100]}...\n")  # 显示前100个字符
                    
                    out_f.write("-" * 40 + "\n")
            
            # 如果是列表类型
            elif isinstance(data, list):
                out_f.write(f"列表长度: {len(data)}\n")
                out_f.write("列表内容预览:\n")
                for i, item in enumerate(data[:5]):  # 显示前5个元素
                    out_f.write(f"索引 {i}: {type(item)} - {str(item)[:100]}\n")
                if len(data) > 5:
                    out_f.write("...\n")
            
            # 如果是NumPy数组
            elif isinstance(data, np.ndarray):
                out_f.write(f"NumPy数组形状: {data.shape}\n")
                out_f.write(f"数据类型: {data.dtype}\n")
                out_f.write(f"数组内容预览: {data.flatten()[:20]}\n")  # 显示前20个元素
            
            # 其他类型
            else:
                out_f.write(f"数据内容: {str(data)[:200]}\n")  # 显示前200个字符
        
        print(f"分析完成！结果已保存到: {output_file}")
        return output_file
            
    except Exception as e:
        error_msg = f"读取文件时出错: {e}"
        print(error_msg)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as out_f:
                out_f.write(error_msg)
        return None

def analyze_pkl_structure(file_path, output_file=None, append_mode=False):
    """
    深度分析pkl文件的结构并保存到txt文件
    
    Args:
        file_path (str): pkl文件的路径
        output_file (str): 输出txt文件的路径
        append_mode (bool): 是否追加到现有文件
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 如果没有指定输出文件，使用与基本分析相同的文件
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{base_name}_analysis_{timestamp}.txt"
        
        # 选择写入模式
        mode = 'a' if append_mode else 'w'
        
        with open(output_file, mode, encoding='utf-8') as out_f:
            if not append_mode:
                out_f.write(f"PKL文件详细结构分析\n")
                out_f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                out_f.write(f"源文件: {file_path}\n")
            
            out_f.write("\n" + "=" * 60 + "\n")
            out_f.write("详细结构分析:\n")
            out_f.write("=" * 60 + "\n")
            
            def print_structure(obj, indent=0, max_depth=3, output_stream=None):
                """递归打印对象结构到文件"""
                if indent > max_depth:
                    output_stream.write("  " * indent + "... (超过最大深度)\n")
                    return
                
                if isinstance(obj, dict):
                    output_stream.write("  " * indent + f"字典 (包含 {len(obj)} 个键):\n")
                    for key, value in list(obj.items())[:10]:  # 最多显示10个键
                        output_stream.write("  " * (indent + 1) + f"'{key}': ")
                        if isinstance(value, (dict, list)):
                            output_stream.write("\n")
                            print_structure(value, indent + 2, max_depth, output_stream)
                        else:
                            output_stream.write(f"{type(value).__name__}")
                            if hasattr(value, 'shape'):
                                output_stream.write(f" {value.shape}")
                            elif hasattr(value, '__len__'):
                                try:
                                    output_stream.write(f" (长度: {len(value)})")
                                except:
                                    pass
                            output_stream.write("\n")
                    if len(obj) > 10:
                        output_stream.write("  " * (indent + 1) + "... (还有更多键)\n")
                
                elif isinstance(obj, list):
                    output_stream.write("  " * indent + f"列表 (包含 {len(obj)} 个元素)\n")
                    if len(obj) > 0:
                        output_stream.write("  " * (indent + 1) + f"元素类型: {type(obj[0]).__name__}\n")
                        if len(obj) <= 5:
                            for i, item in enumerate(obj):
                                output_stream.write("  " * (indent + 1) + f"[{i}]: ")
                                print_structure(item, indent + 2, max_depth, output_stream)
                
                elif isinstance(obj, np.ndarray):
                    output_stream.write("  " * indent + f"NumPy数组: {obj.shape}, dtype: {obj.dtype}\n")
                
                else:
                    output_stream.write("  " * indent + f"{type(obj).__name__}: {str(obj)[:50]}\n")
            
            print_structure(data, output_stream=out_f)
        
        print(f"详细结构分析完成！结果已保存到: {output_file}")
        return output_file
        
    except Exception as e:
        error_msg = f"分析结构时出错: {e}"
        print(error_msg)
        if output_file:
            with open(output_file, 'a' if append_mode else 'w', encoding='utf-8') as out_f:
                out_f.write(f"\n{error_msg}\n")
        return None

def batch_analyze_pkl_files(directory_path, pattern="*.pkl"):
    """
    批量分析目录中的所有pkl文件
    
    Args:
        directory_path (str): 包含pkl文件的目录路径
        pattern (str): 文件匹配模式，默认为"*.pkl"
    """
    import glob
    
    pkl_files = glob.glob(os.path.join(directory_path, pattern))
    
    if not pkl_files:
        print(f"在目录 {directory_path} 中没有找到匹配的pkl文件")
        return
    
    print(f"找到 {len(pkl_files)} 个pkl文件，开始批量分析...")
    
    # 创建汇总文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"pkl_batch_analysis_summary_{timestamp}.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as summary_f:
        summary_f.write(f"PKL文件批量分析汇总报告\n")
        summary_f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_f.write(f"分析目录: {directory_path}\n")
        summary_f.write(f"总文件数: {len(pkl_files)}\n")
        summary_f.write("=" * 80 + "\n\n")
        
        for i, pkl_file in enumerate(pkl_files, 1):
            print(f"正在分析第 {i}/{len(pkl_files)} 个文件: {os.path.basename(pkl_file)}")
            
            # 为每个文件创建单独的分析文件
            individual_output = read_pkl_file(pkl_file)
            
            if individual_output:
                # 将简要信息写入汇总文件
                summary_f.write(f"文件 {i}: {os.path.basename(pkl_file)}\n")
                summary_f.write(f"详细分析文件: {individual_output}\n")
                
                # 读取文件的基本信息添加到汇总
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                    summary_f.write(f"数据类型: {type(data)}\n")
                    if isinstance(data, dict):
                        summary_f.write(f"键的数量: {len(data)}\n")
                        summary_f.write(f"主要键名: {list(data.keys())[:5]}\n")
                    elif isinstance(data, list):
                        summary_f.write(f"列表长度: {len(data)}\n")
                    elif isinstance(data, np.ndarray):
                        summary_f.write(f"数组形状: {data.shape}\n")
                except Exception as e:
                    summary_f.write(f"读取错误: {e}\n")
                
                summary_f.write("-" * 40 + "\n")
            else:
                summary_f.write(f"文件 {i}: {os.path.basename(pkl_file)} - 分析失败\n")
                summary_f.write("-" * 40 + "\n")
    
    print(f"\n批量分析完成！汇总报告保存到: {summary_file}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    file_path = "motion_data/mimic_filtered_v2/g1_priv_mimic-71-randclip-014339_v2.pkl"
    
    print("开始分析PKL文件...")
    
    # 方法1: 分析单个文件
    output_file = read_pkl_file(file_path)
    
    if output_file:
        # 详细结构分析，追加到同一个文件
        analyze_pkl_structure(file_path, output_file, append_mode=True)
        print(f"\n完整分析报告已保存到: {output_file}")
    
    # 方法2: 批量分析整个目录（取消注释以使用）
    # batch_analyze_pkl_files("motion_data/mimic_filtered_v2/")
    
    # 方法3: 如果你想分别保存基本分析和详细分析到不同文件，可以这样调用：
    # basic_output = read_pkl_file(file_path, "basic_analysis.txt")
    # detail_output = analyze_pkl_structure(file_path, "detailed_analysis.txt")
    
    # 方法4: 如果你想读取其他pkl文件，可以这样调用：
    # read_pkl_file("your_other_file.pkl", "output_analysis.txt")import pickle
import numpy as np
import os
from datetime import datetime

def read_pkl_file(file_path, output_file=None):
    """
    读取pkl文件并将键名和键值保存到txt文件
    
    Args:
        file_path (str): pkl文件的路径
        output_file (str): 输出txt文件的路径，如果为None则自动生成
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            error_msg = f"错误: 文件 {file_path} 不存在"
            print(error_msg)
            return
        
        # 生成输出文件名
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{base_name}_analysis_{timestamp}.txt"
        
        # 读取pkl文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 打开输出文件
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(f"PKL文件分析报告\n")
            out_f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            out_f.write(f"源文件: {file_path}\n")
            out_f.write("=" * 60 + "\n\n")
            
            out_f.write(f"成功读取文件: {file_path}\n")
            out_f.write("=" * 60 + "\n")
            
            # 检查数据类型
            out_f.write(f"数据类型: {type(data)}\n")
            out_f.write("=" * 60 + "\n")
        
            # 如果是字典类型，打印所有键值对
            if isinstance(data, dict):
                out_f.write("字典中的键名和对应的键值:\n")
                out_f.write("-" * 40 + "\n")
                
                for key, value in data.items():
                    out_f.write(f"键名: {key}\n")
                    out_f.write(f"键值类型: {type(value)}\n")
                    
                    # 根据值的类型进行不同的处理
                    if isinstance(value, (str, int, float, bool)):
                        out_f.write(f"键值: {value}\n")
                    elif isinstance(value, (list, tuple)):
                        out_f.write(f"键值长度: {len(value)}\n")
                        if len(value) > 0:
                            out_f.write(f"第一个元素: {value[0]} (类型: {type(value[0])})\n")
                            if len(value) > 5:
                                out_f.write(f"前5个元素: {value[:5]}\n")
                            else:
                                out_f.write(f"所有元素: {value}\n")
                    elif isinstance(value, np.ndarray):
                        out_f.write(f"NumPy数组形状: {value.shape}\n")
                        out_f.write(f"数据类型: {value.dtype}\n")
                        if value.size > 0:
                            out_f.write(f"数组内容预览: {value.flatten()[:10]}...\n")  # 显示前10个元素
                    else:
                        out_f.write(f"键值: {str(value)[:100]}...\n")  # 显示前100个字符
                    
                    out_f.write("-" * 40 + "\n")
            
            # 如果是列表类型
            elif isinstance(data, list):
                out_f.write(f"列表长度: {len(data)}\n")
                out_f.write("列表内容预览:\n")
                for i, item in enumerate(data[:5]):  # 显示前5个元素
                    out_f.write(f"索引 {i}: {type(item)} - {str(item)[:100]}\n")
                if len(data) > 5:
                    out_f.write("...\n")
            
            # 如果是NumPy数组
            elif isinstance(data, np.ndarray):
                out_f.write(f"NumPy数组形状: {data.shape}\n")
                out_f.write(f"数据类型: {data.dtype}\n")
                out_f.write(f"数组内容预览: {data.flatten()[:20]}\n")  # 显示前20个元素
            
            # 其他类型
            else:
                out_f.write(f"数据内容: {str(data)[:200]}\n")  # 显示前200个字符
        
        print(f"分析完成！结果已保存到: {output_file}")
        return output_file
            
    except Exception as e:
        error_msg = f"读取文件时出错: {e}"
        print(error_msg)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as out_f:
                out_f.write(error_msg)
        return None

def analyze_pkl_structure(file_path, output_file=None, append_mode=False):
    """
    深度分析pkl文件的结构并保存到txt文件
    
    Args:
        file_path (str): pkl文件的路径
        output_file (str): 输出txt文件的路径
        append_mode (bool): 是否追加到现有文件
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 如果没有指定输出文件，使用与基本分析相同的文件
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{base_name}_analysis_{timestamp}.txt"
        
        # 选择写入模式
        mode = 'a' if append_mode else 'w'
        
        with open(output_file, mode, encoding='utf-8') as out_f:
            if not append_mode:
                out_f.write(f"PKL文件详细结构分析\n")
                out_f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                out_f.write(f"源文件: {file_path}\n")
            
            out_f.write("\n" + "=" * 60 + "\n")
            out_f.write("详细结构分析:\n")
            out_f.write("=" * 60 + "\n")
            
            def print_structure(obj, indent=0, max_depth=3, output_stream=None):
                """递归打印对象结构到文件"""
                if indent > max_depth:
                    output_stream.write("  " * indent + "... (超过最大深度)\n")
                    return
                
                if isinstance(obj, dict):
                    output_stream.write("  " * indent + f"字典 (包含 {len(obj)} 个键):\n")
                    for key, value in list(obj.items())[:10]:  # 最多显示10个键
                        output_stream.write("  " * (indent + 1) + f"'{key}': ")
                        if isinstance(value, (dict, list)):
                            output_stream.write("\n")
                            print_structure(value, indent + 2, max_depth, output_stream)
                        else:
                            output_stream.write(f"{type(value).__name__}")
                            if hasattr(value, 'shape'):
                                output_stream.write(f" {value.shape}")
                            elif hasattr(value, '__len__'):
                                try:
                                    output_stream.write(f" (长度: {len(value)})")
                                except:
                                    pass
                            output_stream.write("\n")
                    if len(obj) > 10:
                        output_stream.write("  " * (indent + 1) + "... (还有更多键)\n")
                
                elif isinstance(obj, list):
                    output_stream.write("  " * indent + f"列表 (包含 {len(obj)} 个元素)\n")
                    if len(obj) > 0:
                        output_stream.write("  " * (indent + 1) + f"元素类型: {type(obj[0]).__name__}\n")
                        if len(obj) <= 5:
                            for i, item in enumerate(obj):
                                output_stream.write("  " * (indent + 1) + f"[{i}]: ")
                                print_structure(item, indent + 2, max_depth, output_stream)
                
                elif isinstance(obj, np.ndarray):
                    output_stream.write("  " * indent + f"NumPy数组: {obj.shape}, dtype: {obj.dtype}\n")
                
                else:
                    output_stream.write("  " * indent + f"{type(obj).__name__}: {str(obj)[:50]}\n")
            
            print_structure(data, output_stream=out_f)
        
        print(f"详细结构分析完成！结果已保存到: {output_file}")
        return output_file
        
    except Exception as e:
        error_msg = f"分析结构时出错: {e}"
        print(error_msg)
        if output_file:
            with open(output_file, 'a' if append_mode else 'w', encoding='utf-8') as out_f:
                out_f.write(f"\n{error_msg}\n")
        return None

def batch_analyze_pkl_files(directory_path, pattern="*.pkl"):
    """
    批量分析目录中的所有pkl文件
    
    Args:
        directory_path (str): 包含pkl文件的目录路径
        pattern (str): 文件匹配模式，默认为"*.pkl"
    """
    import glob
    
    pkl_files = glob.glob(os.path.join(directory_path, pattern))
    
    if not pkl_files:
        print(f"在目录 {directory_path} 中没有找到匹配的pkl文件")
        return
    
    print(f"找到 {len(pkl_files)} 个pkl文件，开始批量分析...")
    
    # 创建汇总文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"pkl_batch_analysis_summary_{timestamp}.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as summary_f:
        summary_f.write(f"PKL文件批量分析汇总报告\n")
        summary_f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_f.write(f"分析目录: {directory_path}\n")
        summary_f.write(f"总文件数: {len(pkl_files)}\n")
        summary_f.write("=" * 80 + "\n\n")
        
        for i, pkl_file in enumerate(pkl_files, 1):
            print(f"正在分析第 {i}/{len(pkl_files)} 个文件: {os.path.basename(pkl_file)}")
            
            # 为每个文件创建单独的分析文件
            individual_output = read_pkl_file(pkl_file)
            
            if individual_output:
                # 将简要信息写入汇总文件
                summary_f.write(f"文件 {i}: {os.path.basename(pkl_file)}\n")
                summary_f.write(f"详细分析文件: {individual_output}\n")
                
                # 读取文件的基本信息添加到汇总
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                    summary_f.write(f"数据类型: {type(data)}\n")
                    if isinstance(data, dict):
                        summary_f.write(f"键的数量: {len(data)}\n")
                        summary_f.write(f"主要键名: {list(data.keys())[:5]}\n")
                    elif isinstance(data, list):
                        summary_f.write(f"列表长度: {len(data)}\n")
                    elif isinstance(data, np.ndarray):
                        summary_f.write(f"数组形状: {data.shape}\n")
                except Exception as e:
                    summary_f.write(f"读取错误: {e}\n")
                
                summary_f.write("-" * 40 + "\n")
            else:
                summary_f.write(f"文件 {i}: {os.path.basename(pkl_file)} - 分析失败\n")
                summary_f.write("-" * 40 + "\n")
    
    print(f"\n批量分析完成！汇总报告保存到: {summary_file}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    file_path = "motion_data/mimic_filtered_v2/g1_priv_mimic-71-randclip-014339_v2.pkl"
    
    print("开始分析PKL文件...")
    
    # 方法1: 分析单个文件
    output_file = read_pkl_file(file_path)
    
    if output_file:
        # 详细结构分析，追加到同一个文件
        analyze_pkl_structure(file_path, output_file, append_mode=True)
        print(f"\n完整分析报告已保存到: {output_file}")
    
    # 方法2: 批量分析整个目录（取消注释以使用）
    # batch_analyze_pkl_files("motion_data/mimic_filtered_v2/")
    
    # 方法3: 如果你想分别保存基本分析和详细分析到不同文件，可以这样调用：
    # basic_output = read_pkl_file(file_path, "basic_analysis.txt")
    # detail_output = analyze_pkl_structure(file_path, "detailed_analysis.txt")
    
    # 方法4: 如果你想读取其他pkl文件，可以这样调用：
    # read_pkl_file("your_other_file.pkl", "output_analysis.txt")