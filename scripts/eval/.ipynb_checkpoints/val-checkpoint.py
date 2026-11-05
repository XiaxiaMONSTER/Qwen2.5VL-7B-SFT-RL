import json
import argparse
import os
import sys
from pathlib import Path
# 获取当前文件的绝对路径，然后定位项目根目录（向上两级）
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  # scripts/val -> scripts -> project_root
sys.path.insert(0, str(root_dir))  # 将根目录添加到模块搜索路径

from src.utils.assessment import get_result, match_answer, match_traffic_lights
from src.utils.config_loader import load_config

def main(model_path, val_path, model_name):
    config = load_config()
    output_folder = config["result"]["model_output"]
    output_json_path = os.path.join(output_folder, f"{model_name}.json")
    get_result(model_path=model_path, val_path=val_path, output_json_path=output_json_path)
    # 加载模型预测结果
    with open(output_json_path, 'r') as f:
        model_result = json.load(f)["output"]

    # 加载验证集的标准答案
    with open(val_path, 'r') as f:
        answer_response = json.load(f)

    # 评估指标存储
    scene_correct = 0
    total_images = 0

    # 交通灯检测指标
    detection_tp = 0  # 真正例（正确检测到的交通灯）
    detection_fp = 0  # 假正例（错误检测到的交通灯）
    detection_fn = 0  # 假反例（未检测到的交通灯）

    # 状态指标
    state_t = 0      # 状态预测正确
    state_f = 0      # 状态预测错误或未预测

    # 方向指示指标
    indications_t = 0
    indications_f = 0

    # 倒计时指标
    countdown_tp = 0
    countdown_fn = 0
    countdown_fp = 0
    countdown_total = 0  # 有倒计时的交通灯总数

    # 遍历模型预测结果
    for item in model_result:
        # 从模型输出中提取预测结果
        model_dict = match_answer(item["response"])
        
        # 获取当前图像的路径
        image_path = item["image_path"]
        total_images += 1
        
        # 在验证集中查找匹配的图像路径
        index = None
        for i, resp in enumerate(answer_response):
            if resp["images"] == image_path:
                index = i
                break
        
        if index is not None:
            # 从验证集中提取标准答案内容
            answer_content = answer_response[index]["messages"][2]["content"]
            
            # 解析标准答案
            answer_dict = match_answer(answer_content)
            
            # 1. 比较场景描述
            if model_dict["scene"] == answer_dict["scene"]:
                scene_correct += 1
            
            # 2. 比较交通灯数据
            pred_lights = model_dict["traffic_light"]
            gt_lights = answer_dict["traffic_light"]

            # 匹配预测和真实的交通灯
            matched_pairs, unmatched_pred, unmatched_gt = match_traffic_lights(pred_lights, gt_lights)
            
            # 更新检测指标
            detection_tp += len(matched_pairs)
            detection_fp += len(unmatched_pred)
            detection_fn += len(unmatched_gt)
            
            # 处理匹配的交通灯对
            for pred_idx, gt_idx in matched_pairs:
                try:
                    pred_light = pred_lights[pred_idx]
                    gt_light = gt_lights[gt_idx]
                    
                    # 3. 比较状态
                    if pred_light["state"] == gt_light["state"]:
                        state_t += 1
                    else:
                        state_f += 1
                    
                    # 4. 比较方向指示（忽略顺序）
                    pred_indications = set(pred_light["indications"])
                    gt_indications = set(gt_light["indications"])
                    if pred_indications == gt_indications:
                        indications_t += 1
                    else:
                        indications_f += 1
                    
                    # 5. 比较倒计时数字
                    # 只在有倒计时的情况下计数
                    for pred_idx, gt_idx in matched_pairs:
                        if gt_light["countdown_number"] is not None or pred_light["countdown_number"] is not None:  # 仅当真实存在倒计时
                            countdown_total += 1
                            if gt_light["countdown_number"] is None and pred_light["countdown_number"] is not None:
                                countdown_fp += 1          
                            elif pred_light["countdown_number"] == gt_light["countdown_number"]:
                                countdown_tp += 1
                            else:
                                countdown_fn += 1     
                except:
                    pass
        
        else:
            print(f"警告: 未找到匹配验证数据的图像: {image_path}")

    # 计算总体指标
    scene_accuracy = scene_correct / total_images

    # 检测指标
    detection_precision = detection_tp / (detection_tp + detection_fp) if (detection_tp + detection_fp) > 0 else 0
    detection_recall = detection_tp / (detection_tp + detection_fn) if (detection_tp + detection_fn) > 0 else 0

    # 状态指标
    state_accuracy = state_t / (state_t+state_f)

    # 方向指示指标
    indications_accuracy = indications_t / (indications_t+indications_f)

    # 倒计时指标
    countdown_precision = countdown_tp / (countdown_tp+countdown_fp) if (countdown_tp+countdown_fp) > 0 else 0
    countdown_recall = countdown_tp / (countdown_tp+countdown_fn) if (countdown_tp+countdown_fn) > 0 else 0

    # 打印评估结果
    print("\n评估结果:")
    print("=" * 50)
    print(f"场景准确率: {scene_accuracy:.4f} ({scene_correct}/{total_images})")
    print("\n交通灯相关性检测:")
    print(f"  查准率: {detection_precision:.4f}, 召回率: {detection_recall:.4f}, F1: {2 * detection_precision * detection_recall / (detection_precision + detection_recall):.4f}")
    print(f"  TP: {detection_tp}, FP: {detection_fp}, FN: {detection_fn}")
    print("\n灯态判断:")
    # print(f"  查准率: {state_accuracy:.4f}, 召回率: {state_recall:.4f}, F1: {2 * state_precision * state_recall / (state_precision + state_recall):.4f}")
    print(f"  准确率:{state_accuracy:.4f}")
    print(f"  正确: {state_t}, 错误: {state_f}")
    print("\n指示方向推理:")
    # print(f"  查准率: {indications_accuracy:.4f}, 召回率: {indications_recall:.4f}, F1: {2 * indications_precision * indications_recall / (indications_precision + indications_recall):.4f}")
    print(f"  准确率: {indications_accuracy:.4f}")
    print(f"  正确: {indications_t}, 错误: {indications_f}")
    print("\n倒计时:")
    print(f"  查准率: {countdown_precision:.4f}, 召回率: {countdown_recall:.4f}, F1: {2 * countdown_precision * countdown_recall / (countdown_precision + countdown_recall+0.0001):.4f}")
    print(f"  正确: {countdown_tp}, 错误: {countdown_fn}, 总数: {countdown_total}")
    print("=" * 50)

    # 保存评估结果
    evaluation_results = {
        "info":{
            "model":model_path,
            "val_dataset":val_path,
            "response":output_json_path
        },
        "scene_accuracy": scene_accuracy,
        "detection": {
            "precision": detection_precision,
            "recall": detection_recall,
            "tp": detection_tp,
            "fp": detection_fp,
            "fn": detection_fn
        },
        "state": {
            "accuracy": state_accuracy,
            "t": state_t,
            "f": state_f
        },
        "indications": {
            "accuracy": indications_accuracy,
            "t": indications_t,
            "f": indications_f
        },
        "countdown": {
            "precision": countdown_precision,
            "recall": countdown_recall,
            "tp": countdown_tp,
            "fp": countdown_fp,
            "fn": countdown_fn,
            "total": countdown_total
        }
    }
    evaluation_results_path = os.path.join(config["result"]["evaluation_result"], f"{model_name}.json")
    os.makedirs(os.path.dirname(evaluation_results_path), exist_ok=True)
    with open(evaluation_results_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 JSON 文件中筛选与 JSONL 中图像路径匹配的条目。")
    parser.add_argument('--model_path', required=True, help="JSONL 文件路径")
    parser.add_argument('--val_path', required=True, help="参考 JSON 文件路径")
    parser.add_argument('--model_name', required=True, help="模型名称")

    args = parser.parse_args()
    main(args.model_path, args.val_path, args.model_name)