import os
import json
import argparse
from collections import defaultdict

from utils.instruction_generator import build_instruction_data
from utils.config_loader import load_config


def main(jsonl_path, instruction_path):
    """
    jsonl_path: swift 随机选择的验证集
    instruction_path: 训练使用的指令数据集
    """
    config = load_config()
    instructions_folder = config["data"]["instructions_folder"]
    
    # 加载验证集的标准答案
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    with open(instruction_path,'r') as f:
        all_data = json.load(f)

    filename_dict = defaultdict(list)
    
    # 预处理all_data: 提取文件名并建立索引
    for item in all_data:
        # 提取文件名并移除后缀
        img_path = item["images"][0]  # 假设这是字符串路径
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filename_dict[filename].append(item)

    eval_data = []
    for item in data:
        # 提取主数据的文件名
        img_path = item["images"][0]["path"]  # 假设这是带"path"键的路径
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # 直接从字典中获取匹配项
        if filename in filename_dict:
            eval_data.extend(filename_dict[filename])
    
    with open(os.path.join(instructions_folder, "eval_dataset.json"), "w") as f:
        json.dump(eval_data, f, indent=2)
    print("生成用于测试的数据集：", os.path.join(instructions_folder, "eval_dataset.json"))
    print("测试方法：python ./scripts/eval/eval.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 JSON 文件中筛选与 JSONL 中图像路径匹配的条目。")
    parser.add_argument('--jsonl_path', default="/mnt/xiayuxuan/Qwen_TF/Project-2_trace_sft_prompt_v2/models/Qwen2.5VL-CoT-Template-Full/v7-20250929-144745/val_dataset.jsonl",help="jsonl地址")
    parser.add_argument('--instruction_path', default="/mnt/xiayuxuan/Qwen_TF/Project-2_trace_sft_prompt_v2/data/instructions/marked-smart_resize-cot_template-42.json", help="用于训练的指令数据集地址")

    args = parser.parse_args()
    main(args.jsonl_path, args.instruction_path)