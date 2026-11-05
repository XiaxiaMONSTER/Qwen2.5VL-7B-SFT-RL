import os
import json
import torch
import numpy as np

import ast
import re

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor  # 改回线程池
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from collections import defaultdict
from .config_loader import load_config

def get_model_output(model_path, eval_path, output_json_path, max_batch_size=2):
    """
    读取JSON文件，执行模型推理，并保存结果
    
    Args:
        model_path (str): 模型路径
        eval_path (str): 输入JSON文件路径
        output_json_path (str): 输出JSON文件路径
    """
    from swift.llm import PtEngine, RequestConfig, InferRequest
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['MAX_PIXELS'] = '1003520'
    
    # 初始化模型引擎
    engine = PtEngine(model_path, max_batch_size=max_batch_size)
    request_config = RequestConfig(max_tokens=4096, temperature=0)
    
    # 读取JSON文件
    with open(eval_path, 'r') as f:
        data = json.load(f)
    
    # 准备推理请求列表
    infer_requests = []
    for item in data:
        # 提取system和user消息
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in item["messages"]
            if msg["role"] in ["system", "user"]
        ]
        
        # 获取图片列表（如果存在）
        images = item.get("images", [])
        
        # 创建推理请求
        infer_requests.append(InferRequest(messages=messages, images=images))
    
    # 批量执行推理
    resp_list = engine.infer(infer_requests, request_config)
    
    # 处理结果
    results = []
    for i, (req, resp) in enumerate(zip(infer_requests, resp_list)):
        # 获取最后一条用户消息
        user_prompt = None
        for msg in reversed(req.messages):
            if msg["role"] == "user":
                user_prompt = msg["content"]
                break
        
        # 提取图片地址
        image_paths = req.images
        
        # 获取模型响应
        response = resp.choices[0].message.content
        
        # 添加到结果集
        results.append({
            "prompt": user_prompt,
            "response": response.replace("<|object_ref_start|>", "").replace("<|object_ref_end|>", "").replace("<|box_start|>","").replace("<|box_end|>", ""),
            "image_path": image_paths,
        })

    output_data = {
        "info":{"model_path":model_path, "eval_path":eval_path},
        "output":results
    }
    
    # 保存结果到JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，共处理{len(results)}条数据")
    print(f"结果已保存到: {output_json_path}")
    
    return results

def evaluate_model(output_json_path, eval_path, model_path, model_name, iou_threshold):
    """
    评估交通灯识别模型性能
    
    参数:
        output_json_path: 模型预测结果文件路径
        eval_path: 验证集标准答案文件路径
        model_path: 模型路径（用于记录）
        model_name: 模型名称（用于保存结果）
    """
    config = load_config()
    # 加载模型预测结果
    with open(output_json_path, 'r') as f:
        model_result = json.load(f)["output"]

    # 加载验证集的标准答案
    with open(eval_path, 'r') as f:
        answer_response = json.load(f)

    # 评估指标存储
    scene_correct = 0
    total_images = 0

    # 交通灯检测指标
    detection_tp = 0  # 真正例
    detection_fp = 0  # 假正例
    detection_fn = 0  # 假反例

    # 状态指标
    state_t = 0      # 状态预测正确
    state_f = 0      # 状态预测错误
    state_confusion = {}  # 混淆矩阵

    # 方向指示指标
    indications_t = 0
    indications_f = 0
    indications_confusion = defaultdict(lambda: defaultdict(int))
    all_directions = set()

    # 倒计时指标
    countdown_tp = 0
    countdown_fn = 0
    countdown_fp = 0
    countdown_total = 0

    # 遍历模型预测结果
    for item in model_result:
        # 提取预测结果
        model_dict = match_answer(item["response"])
        image_path = item["image_path"][0]
        total_images += 1
        
        # 在验证集中查找匹配的图像
        index = None
        for i, resp in enumerate(answer_response):
            if resp["images"][0] == image_path:
                index = i
                break
        
        if index is not None:
            # # 调试打印：查看当前 index 对应的 messages 长度和内容
            # print(f"index: {index}")
            # print(f"messages length: {len(answer_response[index]['messages'])}")
            # print(f"messages content: {answer_response[index]['messages']}")
            # 解析标准答案
            answer_content = answer_response[index]["messages"][2]["content"]
            answer_dict = match_answer(answer_content)

            # 1. 比较场景描述
            if model_dict["scene"] == answer_dict["scene"]:
                scene_correct += 1
            
            # 2. 比较交通灯数据
            pred_lights = model_dict["traffic_light"]
            gt_lights = answer_dict["traffic_light"]

            # 匹配预测和真实的交通灯
            matched_pairs, unmatched_pred, unmatched_gt = match_traffic_lights(pred_lights, gt_lights, iou_threshold=iou_threshold)
            
            # 更新检测指标
            detection_tp += len(matched_pairs)
            detection_fp += len(unmatched_pred)
            detection_fn += len(unmatched_gt)
            
            # 处理匹配的交通灯对
            for pred_idx, gt_idx in matched_pairs:
                try:
                    pred_light = pred_lights[pred_idx]
                    gt_light = gt_lights[gt_idx]
                    
                    # 记录状态混淆矩阵
                    true_state = gt_light["state"]
                    pred_state = pred_light["state"]
                    
                    if true_state not in state_confusion:
                        state_confusion[true_state] = {}
                    state_confusion[true_state][pred_state] = state_confusion[true_state].get(pred_state, 0) + 1
                    
                    # 3. 比较状态
                    if pred_light["state"] == gt_light["state"]:
                        state_t += 1
                    else:
                        state_f += 1
                    
                    # 4. 比较方向指示
                    pred_indications = set(pred_light["indications"])
                    gt_indications = set(gt_light["indications"])
                    
                    true_key = tuple(sorted(gt_indications))
                    pred_key = tuple(sorted(pred_indications))
                    
                    all_directions.add(true_key)
                    all_directions.add(pred_key)
                    indications_confusion[true_key][pred_key] += 1
                    
                    if pred_indications == gt_indications:
                        indications_t += 1
                    else:
                        indications_f += 1
                    
                    # 5. 比较倒计时数字
                    if gt_light["countdown_number"] is not None or pred_light["countdown_number"] is not None:
                        countdown_total += 1
                        if gt_light["countdown_number"] is None and pred_light["countdown_number"] is not None:
                            countdown_fp += 1          
                        elif pred_light["countdown_number"] == gt_light["countdown_number"]:
                            countdown_tp += 1
                        else:
                            countdown_fn += 1
                except Exception as e:
                    print(f"处理交通灯时出错: {e}")
                    continue
        else:
            print(f"警告: 未找到匹配验证数据的图像: {image_path}")

    indications_confusion_json = {}
    for true_key, pred_dict in indications_confusion.items():
        # 将元组转换为排序后的字符串表示
        true_key_str = ", ".join(sorted(true_key))
        indications_confusion_json[true_key_str] = {}
        for pred_key, count in pred_dict.items():
            pred_key_str = ", ".join(sorted(pred_key))
            indications_confusion_json[true_key_str][pred_key_str] = count

    # 计算总体指标
    scene_accuracy = scene_correct / total_images if total_images > 0 else 0
    detection_precision = detection_tp / (detection_tp + detection_fp) if (detection_tp + detection_fp) > 0 else 0
    detection_recall = detection_tp / (detection_tp + detection_fn) if (detection_tp + detection_fn) > 0 else 0
    state_accuracy = state_t / (state_t+state_f) if (state_t+state_f) > 0 else 0
    indications_accuracy = indications_t / (indications_t+indications_f) if (indications_t+indications_f) > 0 else 0
    countdown_precision = countdown_tp / (countdown_tp+countdown_fp) if (countdown_tp+countdown_fp) > 0 else 0
    countdown_recall = countdown_tp / (countdown_tp+countdown_fn) if (countdown_tp+countdown_fn) > 0 else 0

    # 打印评估结果
    print("\n评估结果:")
    print("=" * 50)
    print(f"场景准确率: {scene_accuracy:.4f} ({scene_correct}/{total_images})")
    print("\n交通灯相关性检测:")
    print(f"  查准率: {detection_precision:.4f}, 召回率: {detection_recall:.4f}, F1: {2 * detection_precision * detection_recall / (detection_precision + detection_recall +0.0001):.4f}")
    print(f"  TP: {detection_tp}, FP: {detection_fp}, FN: {detection_fn}")
    print("\n灯态判断:")
    print(f"  准确率:{state_accuracy:.4f}")
    print(f"  正确: {state_t}, 错误: {state_f}")
    
    # 打印状态混淆矩阵
    print("\n灯态判断混淆矩阵:")
    state_order = ['red', 'green', 'yellow', 'off', 'unknown']
    total_by_true_state = {s: sum(pred.values()) for s, pred in state_confusion.items()}
    
    header = "真实状态\\预测状态 | " + " | ".join([f"{state:<10}" for state in state_order]) + " | 总计"
    print(header)
    print("-" * len(header))
    
    for true_state in state_order:
        if true_state not in state_confusion:
            continue
            
        row = f"{true_state:<15} | "
        for pred_state in state_order:
            count = state_confusion[true_state].get(pred_state, 0)
            total = total_by_true_state[true_state]
            percentage = f"({count/total:.1%})" if total > 0 else ""
            row += f"{count:>3} {percentage:<7} | "
        row += f"{total_by_true_state[true_state]:>5}"
        print(row)
    
    col_totals = {state: 0 for state in state_order}
    for pred_counts in state_confusion.values():
        for pred_state, count in pred_counts.items():
            if pred_state in col_totals:
                col_totals[pred_state] += count
    
    total_row = "总计         | "
    for state in state_order:
        total_row += f"{col_totals.get(state, 0):>11} | "
    print("-" * len(header))
    print(total_row)
    
    # 打印方向指示结果
    print("\n指示方向推理:")
    print(f"  整体准确率: {indications_accuracy:.4f}")
    print(f"  正确: {indications_t}, 错误: {indications_f}")
    
    print("\n倒计时:")
    print(f"  查准率: {countdown_precision:.4f}, 召回率: {countdown_recall:.4f}, F1: {2 * countdown_precision * countdown_recall / (countdown_precision + countdown_recall+0.0001):.4f}")
    print(f"  正确: {countdown_tp}, 错误: {countdown_fn}, 总数: {countdown_total}")
    print("=" * 50)

    # 保存评估结果
    evaluation_results = {
        "info": {
            "model": model_path,
            "val_dataset": eval_path,
            "response": output_json_path
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
            "f": state_f,
            "confusion_matrix": state_confusion
        },
        "indications": {
            "accuracy": indications_accuracy,
            "t": indications_t,
            "f": indications_f,
            "confusion_matrix": indications_confusion_json
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
    print("评测数据保存至: ", evaluation_results_path)
    return evaluation_results


def match_answer(input_str):
    """
    解析模型输出，提取场景信息、交通灯数据和灯编号
    """
    # 1. 提取场景信息
    # 同时支持单引号和双引号（更灵活）
    scene_match = re.search(r"当前场景为 ['\"](.+?)['\"]", input_str)
    scene = scene_match.group(1) if scene_match else None
    
    # 2. 提取</answer>标签内的内容（交通灯数据）
    answer_match = re.search(r"<answer>(.+?)</answer>", input_str, re.DOTALL)
    traffic_data = []
    if answer_match:
        answer_content = answer_match.group(1).strip()
        
        # 处理空列表情况
        if answer_content not in ("[]", "[\n]"):
            try:
                # 尝试直接解析为Python对象
                traffic_data = ast.literal_eval(answer_content.strip())
                
                # 确保是列表格式
                if not isinstance(traffic_data, list):
                    traffic_data = [traffic_data]
                    
                # 标准化输出格式
                for item in traffic_data:
                    # 确保有bbox_2d字段
                    if "bbox_2d" not in item:
                        item["bbox_2d"] = None
                        
                    # 处理countdown_number
                    if "countdown_number" in item and item["countdown_number"] is None:
                        item["countdown_number"] = None
            
            except (SyntaxError, ValueError, TypeError) as e:
                # 如果直接解析失败，尝试更宽松的解析
                traffic_data = []
                
                # 尝试匹配JSON数组格式
                pattern = r'\{.*?\}'
                matches = re.findall(pattern, answer_content, re.DOTALL)
                
                for match in matches:
                    try:
                        # 清理可能的问题字符
                        clean_match = match.replace('\n', '').replace('\\', '')
                        # 添加缺失的引号
                        clean_match = re.sub(r'(\w+):', r'"\1":', clean_match)
                        # 尝试解析
                        item = json.loads(clean_match)
                        traffic_data.append(item)
                    except:
                        print("解析失败")
    
    # 3. 提取整个输入中的Step 3内容（获取交通灯编号）
    # 从整个input_str中截取# Step 3:之后的内容，不受</think>标签限制
    step3_pattern = r'# Step 2:.*?(?=# Step |$)'  # 根据实际回答调整
    step3_match = re.search(step3_pattern, input_str, re.DOTALL)
    step3_content = step3_match.group() if step3_match else ""
    
    # 提取Traffic Light编号
    tl_number_pattern = r'- Traffic Light (\d+)'
    traffic_light_numbers = [
        int(m.group(1)) for m in re.finditer(tl_number_pattern, step3_content)
    ]
    # 4. 构建返回结果
    return {
        "scene": scene,
        "traffic_light": traffic_data,
        "traffic_light_numbers": traffic_light_numbers  # 新字段：交通灯编号列表
    }

def calculate_iou(boxA, boxB):
    """计算两个边界框的IoU（交并比）"""
    # 解包坐标
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    
    # 计算交集区域
    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)
    
    # 计算交集面积
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 计算并集面积
    boxA_area = (xA2 - xA1) * (yA2 - yA1)
    boxB_area = (xB2 - xB1) * (yB2 - yB1)
    union_area = boxA_area + boxB_area - inter_area
    
    # 避免除以零
    return inter_area / union_area if union_area > 0 else 0

def match_traffic_lights(pred_list, gt_list, iou_threshold=1):
    """匹配预测和真实的交通灯，基于IoU，处理空列表情况"""
    # 处理空列表情况
    if not pred_list and not gt_list:
        return [], [], []  # 没有预测也没有真实值
    
    if not pred_list:
        return [], [], list(range(len(gt_list)))  # 没有预测，所有真实值都是未匹配
    
    if not gt_list:
        return [], list(range(len(pred_list))), []  # 没有真实值，所有预测都是未匹配
    
    # 初始化匹配结果
    matched_pairs = []
    unmatched_pred = list(range(len(pred_list)))
    unmatched_gt = list(range(len(gt_list)))
    
    # 计算所有可能的配对
    iou_matrix = np.zeros((len(pred_list), len(gt_list)))
    for i, pred in enumerate(pred_list):
        for j, gt in enumerate(gt_list):
            iou_matrix[i, j] = calculate_iou(pred["bbox_2d"], gt["bbox_2d"])
    
    # 贪心匹配：先匹配IoU最高的对
    while True:
        max_iou = np.max(iou_matrix)
        if max_iou < iou_threshold:
            break
            
        i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        matched_pairs.append((i, j))
        
        # 将已匹配的行和列设为-1
        iou_matrix[i, :] = -1
        iou_matrix[:, j] = -1
        
        # 从未匹配列表中移除
        if i in unmatched_pred:
            unmatched_pred.remove(i)
        if j in unmatched_gt:
            unmatched_gt.remove(j)
    
    return matched_pairs, unmatched_pred, unmatched_gt