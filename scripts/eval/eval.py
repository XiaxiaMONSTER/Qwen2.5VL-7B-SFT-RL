import json
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

# 获取当前文件的绝对路径，然后定位项目根目录（向上两级）
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

from src.utils.assessment import get_model_output, evaluate_model
from src.utils.config_loader import load_config

def main(model_path, eval_path, model_name, max_batch_size, iou_threshold):
    config = load_config()
    output_folder = config["result"]["model_output"]
    
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"已创建输出文件夹: {output_folder}")
    
    output_json_path = os.path.join(output_folder, f"{model_name}.json")
    if os.path.exists(output_json_path):
        print("存在模型输出记录文件：", output_json_path)
        print("直接计算指标，若需要使用模型推理请修改model_name")
    else:
        get_model_output(model_path=model_path, eval_path=eval_path, output_json_path=output_json_path, max_batch_size=max_batch_size)

    # 对模型输出进行评价
    evaluate_model(output_json_path, eval_path, model_path, model_name, iou_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 JSON 文件中筛选与 JSONL 中图像路径匹配的条目。")
    parser.add_argument('--model_path', default="/mnt/xiayuxuan/Qwen_TF/Project-2_trace_sft_prompt_v2/models/Qwen2.5VL-CoT-Template-Full/v7-20250929-144745/checkpoint-4700", help="模型地址")
    parser.add_argument('--eval_path', default="/mnt/xiayuxuan/Qwen_TF/Project-2_trace_sft_prompt_v2/data/instructions/eval_dataset.json", help="测试集地址")
    parser.add_argument('--model_name', required=True, help="模型名称")
    parser.add_argument('--max_batch_size', required=False, default=2, type=int, help="最大批量处理数量")
    parser.add_argument('--iou_threshold', required=False, default=0.99, type=float, help="IoU阈值")

    args = parser.parse_args()
    main(args.model_path, args.eval_path, args.model_name, args.max_batch_size, args.iou_threshold) 