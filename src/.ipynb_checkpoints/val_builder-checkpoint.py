import os
import json
import argparse

from utils.instruction_generator import build_instruction_data
from utils.config_loader import load_config

def main(jsonl_path, json_path):
    """
    jsonl_path: swift随机选择的验证集
    json_path: 全部数据
    """
    config = load_config()
    instructions_folder = config["data"]["instructions_folder"]
    
    # 加载验证集的标准答案
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    with open(json_path,'r') as f:
        all_data = json.load(f)

    val_data = []
    for item in data:
        image_path = item["images"][0]["path"]
        for item in all_data:
            if os.path.splitext(os.path.basename(item["images"]))[0] == os.path.splitext(os.path.basename(image_path))[0]:
                val_data.append(item)
    
    with open(os.path.join(instructions_folder, "val_dataset.json"), "w") as f:
        json.dump(val_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 JSON 文件中筛选与 JSONL 中图像路径匹配的条目。")
    parser.add_argument('--jsonl_path', required=True, help="JSONL文件路径")
    parser.add_argument('--json_path', required=True, help="参考 JSON 文件路径")

    args = parser.parse_args()
    main(args.jsonl_path, args.json_path)