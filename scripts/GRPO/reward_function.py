import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional

import json
import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms, rm_plugins
from swift.utils import get_logger

import numpy as np
from scipy.optimize import linear_sum_assignment
import ast

logger = get_logger()

class FormatReward(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        answer_pattern = re.compile(
            r'<think>.*?</think>\s*<answer>\s*(.*?)\s*</answer>', 
            re.DOTALL
        )
        VALID_STATES = {'red', 'yellow', 'green', 'off', 'unknown'}
        
        for content in completions:
            try:
                # 1. 检查标签结构
                match = answer_pattern.search(content)
                if not match:
                    rewards.append(0.0)
                    continue

                # 2. 解析 JSON - 添加容错处理
                json_str = match.group(1)
                
                try:
                    # 先尝试标准 JSON 解析
                    json_content = json.loads(json_str)
                except json.JSONDecodeError as e:
                    try:
                        # 尝试替换单引号为双引号
                        fixed_json = re.sub(
                            r"'(?![^']*'\s*:)", 
                            '"', 
                            json_str
                        )
                        json_content = json.loads(fixed_json)
                    except:
                        # 尝试使用 literal_eval
                        import ast
                        json_content = ast.literal_eval(json_str)
                
                # 3. 验证数组格式
                if not isinstance(json_content, list):
                    raise ValueError("JSON 应为数组")
                
                # 4. 验证每个对象的字段
                for item in json_content:
                    # 检查必需字段
                    required_fields = {'bbox_2d', 'indications', 'state', 'countdown_number'}
                    if not required_fields.issubset(item.keys()):
                        missing = required_fields - set(item.keys())
                        raise ValueError(f"缺少字段: {', '.join(missing)}")
                    
                    # 验证 bbox_2d 格式
                    bbox = item['bbox_2d']
                    if (
                        not isinstance(bbox, list) or 
                        len(bbox) != 4 or 
                        not all(isinstance(x, int) for x in bbox)
                    ):
                        raise ValueError("bbox_2d 格式错误: 需为4个整数的列表")
                    
                    # 验证 indications 格式
                    indications = item['indications']
                    if (
                        not isinstance(indications, list) or 
                        not all(isinstance(x, str) for x in indications)
                    ):
                        raise ValueError("indications 格式错误: 需为字符串列表")
                    
                    # 验证 state 值
                    state = item['state']
                    if state not in VALID_STATES:
                        raise ValueError(f"非法 state 值: '{state}'")
                    
                    # 验证 countdown_number 值
                    countdown = item['countdown_number']
                    if countdown is not None and not isinstance(countdown, int):
                        raise ValueError("countdown_number 需为整数或 null")
                
                rewards.append(1.0)
            except Exception as e:
                logger.warning(f"格式验证失败: {str(e)}")
                logger.debug(f"问题内容: {json_str}")  # 添加详细日志
                rewards.append(0.0)
        return rewards
    
class BboxReward(ORM):
    def __init__(self, f1_beta=0.5):
        """
        :param f1_beta: Fβ分数中的β值，控制召回率权重
            默认β=0.5（重视精确率）
        """
        self.f1_beta = f1_beta
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        
        for content, sol in zip(completions, solution):
            try:
                # 1. 解析预测和真实框
                pred_items = parse_string(content)
                gt_items = parse_string(sol)
                pred_boxes = [item["bbox_2d"] for item in pred_items]
                gt_boxes = [item["bbox_2d"] for item in gt_items]
                
                # 2. 处理空框特殊情况
                if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                    rewards.append(1.0)  # 完美匹配
                    continue
                    
                if len(gt_boxes) == 0:
                    rewards.append(0.0)  # 无真实框但有预测框
                    continue
                    
                if len(pred_boxes) == 0:
                    rewards.append(0.0)  # 有真实框但无预测框
                    continue
                
                # 3. 匹配框（统一使用0.5阈值）
                matched_pairs, unmatched_gts, unmatched_preds = match_bboxes(
                    gt_boxes, pred_boxes, iou_threshold=0.5
                )
                
                # 4. 计算Fβ分数
                TP = len(matched_pairs)
                FP = len(unmatched_preds)
                FN = len(unmatched_gts)
                
                # 计算加权Fβ分数
                beta_sq = self.f1_beta ** 2
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                
                if precision + recall == 0:
                    f_score = 0.0
                else:
                    f_score = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
                
                # 5. 计算质量系数（平均IoU）
                quality_coeff = 1.0
                if TP > 0:
                    # 计算匹配框的平均IoU
                    total_iou = sum(iou_bbox(gt_boxes[i], pred_boxes[j]) 
                              for i, j in matched_pairs)
                    quality_coeff = total_iou / TP
                
                # 6. 最终奖励 = 质量系数 × Fβ分数
                reward = quality_coeff * f_score
                rewards.append(min(1.0, max(0.0, reward)))
                
            except Exception as e:
                logger.warning(f"计算bbox奖励时出错: {str(e)}")
                rewards.append(0.0)
                
        return rewards


class StateReward(ORM):
#  | 真实状态 | 预测状态 | 得分 | 说明 |
#  |----------|----------|------|------|
#  | 非unknown | 相同状态 | 1.0 | 正确预测 |
#  | 非unknown | unknown | -0.5 | 不必要unknown，强烈惩罚 |
#  | 非unknown | 错误状态 | 0.0 | 错误预测 |
#  | unknown | 非unknown | 0.0 | 错误预测（应输出unknown）|
#  | unknown | unknown | 1.0 | 正确预测 |
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        
        for content, sol in zip(completions, solution):
            try:
                # 1. 解析预测和真实数据
                pred_items = parse_string(content)
                gt_items = parse_string(sol)
                
                # 2. 特殊情况处理：无目标
                if len(gt_items) == 0:
                    # 无真实目标时，预测也应为空
                    reward = 1.0 if len(pred_items) == 0 else 0.0
                    rewards.append(reward)
                    continue
                
                # 3. 匹配框（仅用于关联状态）
                pred_boxes = [item["bbox_2d"] for item in pred_items]
                gt_boxes = [item["bbox_2d"] for item in gt_items]
                matched_pairs, _, _ = match_bboxes(
                    gt_boxes, pred_boxes, iou_threshold=0.5
                )
                
                # 4. 计算状态分数
                total_score = 0.0
                valid_matches = len(matched_pairs)
                
                if valid_matches == 0:
                    rewards.append(0.0)
                    continue
                
                for gt_idx, pred_idx in matched_pairs:
                    gt_state = gt_items[gt_idx]["state"]
                    pred_state = pred_items[pred_idx]["state"]
                    
                    # 状态匹配规则
                    if pred_state == gt_state:
                        # 正确预测：加1分
                        total_score += 1.0
                    elif gt_state != "unknown" and pred_state == "unknown":
                        # 不必要unknown：减0.5分（强烈惩罚）
                        total_score -= 0.5
                    else:
                        # 其他错误：加0分
                        total_score += 0.0
                
                # 5. 计算平均奖励（确保非负）
                reward = total_score / valid_matches
                rewards.append(max(0.0, min(1.0, reward)))  # 裁剪到[0,1]范围
                
            except Exception as e:
                logger.warning(f"计算state奖励时出错: {str(e)}")
                rewards.append(0.0)
                
        return rewards

class IndicationsReward(ORM):
    def __init__(self, unknown_penalty=0.5):
        """
        :param unknown_penalty: 对不必要unknown预测的惩罚系数
        """
        self.unknown_penalty = unknown_penalty
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        
        for content, sol in zip(completions, solution):
            try:
                # 1. 解析预测和真实数据
                pred_items = parse_string(content)
                gt_items = parse_string(sol)
                
                # 2. 处理空列表情况
                if len(gt_items) == 0:
                    # 没有真实目标时，预测也应为空
                    reward = 1.0 if len(pred_items) == 0 else 0.0
                    rewards.append(reward)
                    continue
                
                # 3. 匹配框（仅用于关联指示方向）
                pred_boxes = [item["bbox_2d"] for item in pred_items]
                gt_boxes = [item["bbox_2d"] for item in gt_items]
                matched_pairs, _, _ = match_bboxes(
                    gt_boxes, pred_boxes, iou_threshold=0.5
                )
                
                # 4. 计算指示方向分数
                total_score = 0.0
                valid_matches = len(matched_pairs)
                
                if valid_matches == 0:
                    rewards.append(0.0)
                    continue
                
                for gt_idx, pred_idx in matched_pairs:
                    pred_item = pred_items[pred_idx]
                    gt_item = gt_items[gt_idx]
                    
                    # 获取指示方向列表（不过滤任何值）
                    pred_inds = pred_item["indications"]
                    gt_inds = gt_item["indications"]
                    
                    # 5. 计算Jaccard相似度（考虑所有值）
                    if not gt_inds and not pred_inds:
                        # 双方都没有指示方向
                        score = 1.0
                    elif not gt_inds or not pred_inds:
                        # 一方有指示方向而另一方没有
                        score = 0.0
                    else:
                        # 计算Jaccard相似度
                        gt_set = set(gt_inds)
                        pred_set = set(pred_inds)
                        intersection = gt_set & pred_set
                        union = gt_set | pred_set
                        score = len(intersection) / len(union) if union else 0.0
                    
                    # 6. 检查不必要unknown
                    if any(ind == "unknown" for ind in pred_inds) and \
                       not any(ind == "unknown" for ind in gt_inds):
                        # 存在不必要unknown：应用惩罚
                        score = max(0, score - self.unknown_penalty)
                    
                    total_score += score
                
                # 7. 计算平均奖励
                reward = total_score / valid_matches
                rewards.append(min(1.0, max(0.0, reward)))
                
            except Exception as e:
                logger.warning(f"计算 indications 奖励时出错: {str(e)}")
                rewards.append(0.0)
                
        return rewards

class CountdownReward(ORM):
    def __init__(self, existence_weight=0.7, value_weight=0.3):
        """
        :param existence_weight: 存在性识别的权重
        :param value_weight: 数值准确性的权重
        """
        self.existence_weight = existence_weight
        self.value_weight = value_weight
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        
        for content, sol in zip(completions, solution):
            try:
                # 1. 解析预测和真实数据
                pred_items = parse_string(content)
                gt_items = parse_string(sol)
                
                # 2. 处理空列表情况
                if len(gt_items) == 0:
                    reward = 1.0 if len(pred_items) == 0 else 0.0
                    rewards.append(reward)
                    continue
                
                # 3. 匹配框（仅用于关联倒计时）
                pred_boxes = [item["bbox_2d"] for item in pred_items]
                gt_boxes = [item["bbox_2d"] for item in gt_items]
                matched_pairs, _, _ = match_bboxes(
                    gt_boxes, pred_boxes, iou_threshold=0.5
                )
                
                # 4. 初始化统计量
                tp_exist = 0  # 存在性正确
                fp_exist = 0  # 存在性误报
                fn_exist = 0  # 存在性漏报
                
                tp_value = 0  # 数值正确
                fp_value = 0  # 数值错误
                fn_value = 0  # 数值漏报
                
                # 5. 对每个匹配对进行评估
                for gt_idx, pred_idx in matched_pairs:
                    gt_item = gt_items[gt_idx]
                    pred_item = pred_items[pred_idx]
                    
                    gt_countdown = gt_item["countdown_number"]
                    pred_countdown = pred_item["countdown_number"]
                    
                    # 存在性评估
                    gt_exists = gt_countdown is not None
                    pred_exists = pred_countdown is not None
                    
                    if gt_exists and pred_exists:
                        tp_exist += 1
                        # 数值评估
                        if gt_countdown == pred_countdown:
                            tp_value += 1
                        else:
                            fp_value += 1
                    elif gt_exists and not pred_exists:
                        fn_exist += 1
                        fn_value += 1
                    elif not gt_exists and pred_exists:
                        fp_exist += 1
                        fp_value += 1
                    else:  # 都不存在
                        tp_exist += 1
                        tp_value += 1  # 视为数值正确
                
                # 6. 计算存在性F1分数
                precision_exist = tp_exist / (tp_exist + fp_exist) if (tp_exist + fp_exist) > 0 else 0
                recall_exist = tp_exist / (tp_exist + fn_exist) if (tp_exist + fn_exist) > 0 else 0
                
                if precision_exist + recall_exist == 0:
                    f1_exist = 0
                else:
                    f1_exist = 2 * (precision_exist * recall_exist) / (precision_exist + recall_exist)
                
                # 7. 计算数值F1分数
                precision_value = tp_value / (tp_value + fp_value) if (tp_value + fp_value) > 0 else 0
                recall_value = tp_value / (tp_value + fn_value) if (tp_value + fn_value) > 0 else 0
                
                if precision_value + recall_value == 0:
                    f1_value = 0
                else:
                    f1_value = 2 * (precision_value * recall_value) / (precision_value + recall_value)
                
                # 8. 组合奖励
                reward = (self.existence_weight * f1_exist) + (self.value_weight * f1_value)
                rewards.append(min(1.0, max(0.0, reward)))
                
            except Exception as e:
                logger.warning(f"计算 countdown 奖励时出错: {str(e)}")
                rewards.append(0.0)
                
        return rewards

def match_bboxes(gt_boxes, pred_boxes, iou_threshold=1.0):
    """
    使用匈牙利算法匹配 gt 和 pred 框，使得 IoU 最大化
    :param gt_boxes: list of ground truth boxes, each is [x1, y1, x2, y2]
    :param pred_boxes: list of predicted boxes, each is [x1, y1, x2, y2]
    :param iou_threshold: IoU 阈值，低于该值不认为匹配成功
    :return: matched_pairs, unmatched_gts, unmatched_preds
    """
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)

    if num_gt == 0 or num_pred == 0:
        return [], list(range(num_gt)), list(range(num_pred))

    # 构造 IoU 成本矩阵：负 IoU，因为 linear_sum_assignment 是最小化成本
    cost_matrix = np.zeros((num_gt, num_pred))
    for i in range(num_gt):
        for j in range(num_pred):
            iou = iou_bbox(gt_boxes[i], pred_boxes[j])
            cost_matrix[i][j] = -iou  # 转换为最小化问题

    # 匈牙利算法匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < -iou_threshold:  # 即 IoU > threshold
            matched_pairs.append((r, c))

    # 找出未匹配的索引
    matched_gt = {r for r, _ in matched_pairs}
    matched_pred = {c for _, c in matched_pairs}

    unmatched_gts = [i for i in range(num_gt) if i not in matched_gt]
    unmatched_preds = [i for i in range(num_pred) if i not in matched_pred]

    return matched_pairs, unmatched_gts, unmatched_preds

def iou_bbox(box1, box2):
    """
    计算两个 bbox 的 IoU
    box: [x1, y1, x2, y2] 左上角和右下角坐标
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # 计算交集区域的坐标
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    # 计算交集区域面积
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算各自面积
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)

    # 并集面积
    union_area = area1 + area2 - inter_area

    # 防止除以零
    if union_area == 0:
        return 0.0

    return inter_area / union_area

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



orms['external_format'] = FormatReward
# orms['external_bbox_end2end'] = BboxReward(is_end_to_end=True)
orms['external_bbox'] = BboxReward
orms['external_state'] = StateReward
orms['external_indication'] = IndicationsReward
orms['external_countdown'] = CountdownReward