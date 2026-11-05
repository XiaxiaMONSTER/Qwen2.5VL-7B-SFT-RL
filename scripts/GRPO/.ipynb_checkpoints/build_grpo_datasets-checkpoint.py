import json
import re
import ast
import argparse
import sys
import os
from pathlib import Path

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

def build_grpo_datasets(json_path):
    """
    处理JSON数据并构建新数据集
    :param json_path: 包含原始数据的JSON文件路径
    :return: 处理后的数据集列表
    """
    # 加载原始数据
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    processed_data = []
    
    for item in raw_data:
        try:
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
                "images": [item["images"]],  # 注意：包装成列表
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
            continue
    
    return processed_data

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='构建 GRPO 数据集')
    config = load_config()
    parser.add_argument('-i', '--input', type=str, 
                        default="./data/instructions/marked-fixed_ratio-cot_template.json",
                        help='输入文件路径 (默认: %(default)s)')
    
    parser.add_argument('-o', '--output', type=str, default="grpo_datasets.json",
                        help='输出文件路径 (默认: %(default)s)')

    
    # 解析命令行参数
    args = parser.parse_args()
    output_path = os.path.join(config["data"]["instructions_folder"], args.output)
    dataset = build_grpo_datasets(args.input)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)