import json
import re
import ast
import argparse
import sys
import os
from pathlib import Path
from PIL import Image
import shutil

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

from src.utils.config_loader import load_config

def parse_string(s):
    # 改进的正则表达式：只捕获 <answer> 标签内的内容（不包括标签本身）
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    match = answer_pattern.search(s)
    
    if not match:
        raise ValueError("未找到<answer>标签")
    
    # 提取标签内的内容并清理空白字符
    json_str = match.group(1).strip()
    
    # 处理空列表情况（如 <answer>[]</answer>）
    if json_str == "[]":
        return []
    
    try:
        # 尝试标准 JSON 解析
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 尝试解析 Python 风格的单引号格式
            return ast.literal_eval(json_str)
        except Exception as e:
            # 处理特殊格式（如带有换行符的列表）
            try:
                # 移除换行符并再次尝试解析
                cleaned_str = json_str.replace("\n", "")
                return ast.literal_eval(cleaned_str)
            except:
                raise ValueError(f"JSON 解析失败: {e}\n内容: {json_str}")

def check_image_size(image_path, max_dimension=2500):
    """
    检查图像尺寸是否超过阈值
    :param image_path: 图像文件路径
    :param max_dimension: 允许的最大尺寸（宽或高）
    :return: True如果图像尺寸过大，False如果正常
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width > max_dimension or height > max_dimension:
                return True
    except Exception as e:
        print(f"⚠️ 无法检查图像尺寸: {image_path} - {str(e)}")
    return False

def build_grpo_datasets(json_path, max_dimension=2500, backup_dir=None):
    """
    处理JSON数据并构建新数据集，同时过滤超大尺寸图像
    :param json_path: 包含原始数据的JSON文件路径
    :param max_dimension: 允许的最大图像尺寸（默认2500像素）
    :param backup_dir: 备份超大图像的目录（可选）
    :return: 处理后的数据集列表
    """
    # 加载原始数据
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    # 获取JSON文件所在目录作为基础路径
    base_dir = os.path.dirname(os.path.abspath(json_path))
    
    # 创建备份目录（如果需要）
    if backup_dir and not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    processed_data = []
    oversized_count = 0
    skipped_count = 0
    
    print(f"开始处理数据集，过滤超大图像（阈值: {max_dimension}像素）...")
    for item in raw_data:
        try:
            # 获取图像路径（注意：原始数据中的images字段是字符串）
            image_path = item["images"][0]
            
            
            # 检查图像是否存在
            if not os.path.exists(image_path):
                print(f"⚠️ 图像文件不存在: {image_path}")
                skipped_count += 1
                continue
                
            # 检查图像尺寸
            if check_image_size(image_path, max_dimension):
                oversized_count += 1
                
                # 备份超大图像（如果需要）
                if backup_dir:
                    backup_path = os.path.join(backup_dir, os.path.basename(image_path))
                    shutil.copy2(image_path, backup_path)
                    print(f"📦 备份超大图像: {os.path.basename(image_path)} -> {backup_path}")
                
                # 跳过超大图像样本
                continue
            
            # 提取用户消息
            user_message = next(
                msg['content'] for msg in item['messages'] 
                if msg['role'] == 'user'
            )
            
            # 提取助手消息
            assistant_message = next(
                msg['content'] for msg in item['messages'] 
                if msg['role'] == 'assistant'
            )
            
            # 解析solution
            parse_string(assistant_message)  # 验证格式
            answer_match = re.search(r'(<answer>.*?</answer>)', assistant_message, re.DOTALL)
            if not answer_match:
                raise ValueError("未找到<answer>标签")
            solution = answer_match.group(1).strip()
            
            # 构建新数据项
            new_item = {
                "images": [image_path],  # 注意：包装成列表
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "solution": solution
            }
            processed_data.append(new_item)
            
        except (StopIteration, ValueError, KeyError) as e:
            print(f"跳过无效数据项: {e}")
            skipped_count += 1
            continue
    
    # 打印统计信息
    print("\n" + "="*50)
    print("数据集构建报告")
    print("="*50)
    print(f"原始样本数: {len(raw_data)}")
    print(f"处理后的样本数: {len(processed_data)}")
    print(f"跳过的超大图像样本: {oversized_count}")
    print(f"其他原因跳过的样本: {skipped_count}")
    
    if backup_dir:
        print(f"备份的超大图像数: {oversized_count}")
        print(f"备份目录: {os.path.abspath(backup_dir)}")
    
    return processed_data

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='构建 GRPO 数据集并过滤超大图像')
    config = load_config()
    parser.add_argument('-i', '--instruction_path', type=str, 
                        default="/root/autodl-tmp/traffic_lights/data/instructions/marked-smart_resize-cot_template-42.json",
                        help='输入文件路径 (默认: %(default)s)')
    
    parser.add_argument('-o', '--output', type=str, default="grpo_datasets.json",
                        help='输出文件路径 (默认: %(default)s)')
    
    # 添加新参数
    parser.add_argument('--max_dimension', type=int, default=2500,
                        help='允许的最大图像尺寸（默认2500像素）')
    parser.add_argument('--backup_dir', type=str, 
                        help='备份超大图像的目录（可选）')
    
    # 解析命令行参数
    args = parser.parse_args()
    output_path = os.path.join(config["data"]["instructions_folder"], args.output)
    
    # 构建数据集
    dataset = build_grpo_datasets(
        args.instruction_path,
        max_dimension=args.max_dimension,
        backup_dir=args.backup_dir
    )
    
    # 保存处理后的数据集
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ GRPO数据集已保存至: {os.path.abspath(output_path)}")