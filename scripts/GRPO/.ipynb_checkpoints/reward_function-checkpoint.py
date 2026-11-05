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
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        alpha = 0.25  # 漏检惩罚系数
        beta = 0.25   # 误检惩罚系数
        
        for content, sol in zip(completions, solution):
            try:
                # 1. 解析预测和真实框
                pred_boxes = [item["bbox_2d"] for item in parse_string(content)]
                gt_boxes = [item["bbox_2d"] for item in parse_string(sol)]
                
                # 2. 处理空框特殊情况
                if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                    rewards.append(1.0)  # 双方都无框，完美匹配
                    continue
                    
                if len(gt_boxes) == 0:
                    # 无真实框但有预测框：纯误检惩罚
                    rewards.append(max(0.0, -beta))
                    continue
                    
                if len(pred_boxes) == 0:
                    # 有真实框但无预测框：纯漏检惩罚
                    rewards.append(max(0.0, -alpha))
                    continue
                
                # 3. 匹配框并计算基础IoU奖励
                matched_pairs, unmatched_gts, unmatched_preds = match_bboxes(
                    gt_boxes, pred_boxes, iou_threshold=0.5
                )
                
                # 计算匹配框的平均IoU（避免除零）
                iou_reward = 0.0
                if matched_pairs:
                    total_iou = sum(iou_bbox(gt_boxes[i], pred_boxes[j]) 
                                 for i, j in matched_pairs)
                    iou_reward = total_iou / len(matched_pairs)
                
                # 4. 计算惩罚项（基于比例）
                miss_penalty = -alpha * (len(unmatched_gts) / len(gt_boxes))
                extra_penalty = -beta * (len(unmatched_preds) / len(pred_boxes))
                
                # 5. 组合奖励（确保范围合理）
                reward_all = iou_reward + miss_penalty + extra_penalty
                rewards.append(max(0.0, min(1.0, reward_all)))
                
            except Exception as e:
                logger.warning(f"计算bbox奖励时出错: {str(e)}")
                rewards.append(0.0)
                
        return rewards

class StateReward(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solution):
            try:
                # 1. 解析预测和真实数据
                pred_items = parse_string(content)  # 获取完整项目列表
                gt_items = parse_string(sol)       # 获取完整项目列表
                
                # 提取框和状态（保持关联）
                pred_boxes = [item["bbox_2d"] for item in pred_items]
                gt_boxes = [item["bbox_2d"] for item in gt_items]
                pred_states = [item["state"] for item in pred_items]
                gt_states = [item["state"] for item in gt_items]
                
                # 2. 匹配框
                matched_pairs, unmatched_gts, unmatched_preds = match_bboxes(
                    gt_boxes, pred_boxes, iou_threshold=0.5
                )
                
                # 3. 计算匹配框的状态正确率
                correct_states = 0
                for gt_idx, pred_idx in matched_pairs:
                    if pred_states[pred_idx] == gt_states[gt_idx]:
                        correct_states += 1
                
                # 4. 计算状态奖励（考虑匹配和未匹配）
                total_boxes = len(gt_boxes)  # 以真实框为基准
                if total_boxes == 0:
                    # 没有真实框的特殊情况
                    reward = 1.0 if len(pred_boxes) == 0 else 0.0
                else:
                    # 基础奖励：匹配框中的正确状态比例
                    match_reward = correct_states / len(matched_pairs) if matched_pairs else 0
                    
                    # 惩罚：漏检框（状态未识别）和误检框（错误状态）
                    miss_penalty = 0.5 * (len(unmatched_gts) / total_boxes)
                    extra_penalty = 0.3 * (len(unmatched_preds) / total_boxes)
                    
                    # 组合奖励（匹配奖励占主导）
                    reward = match_reward - miss_penalty - extra_penalty
                
                # 确保奖励在 [0,1] 范围内
                rewards.append(max(0.0, min(1.0, reward)))
                
            except Exception as e:
                logger.warning(f"计算state奖励时出错: {str(e)}")
                rewards.append(0.0)
                
        return rewards

class IndicationsReward(ORM):
    """
    专门评估 indications 属性的奖励函数
    - 过滤掉 "NA" 和 "unknown" 值
    - 计算匹配框的 Jaccard 相似度
    - 考虑有效指示的存在性
    - 处理空列表情况
    """
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
                
                # 3. 匹配框（基于bbox_2d）
                pred_boxes = [item["bbox_2d"] for item in pred_items]
                gt_boxes = [item["bbox_2d"] for item in gt_items]
                matched_pairs, unmatched_gts, unmatched_preds = match_bboxes(
                    gt_boxes, pred_boxes, iou_threshold=0.5
                )
                
                # 4. 初始化奖励
                total_score = 0.0
                valid_pairs = 0
                
                # 5. 对每个匹配对计算 indications 奖励
                for gt_idx, pred_idx in matched_pairs:
                    pred_item = pred_items[pred_idx]
                    gt_item = gt_items[gt_idx]
                    
                    # 过滤无效值
                    pred_indications = [i for i in pred_item["indications"] 
                                        if i not in ["NA", "unknown"]]
                    gt_indications = [i for i in gt_item["indications"] 
                                      if i not in ["NA", "unknown"]]
                    
                    # 计算相似度
                    if gt_indications and pred_indications:
                        # 计算 Jaccard 相似度
                        intersection = len(set(gt_indications) & set(pred_indications))
                        union = len(set(gt_indications) | set(pred_indications))
                        score = intersection / union
                    elif not gt_indications and not pred_indications:
                        # 双方都没有有效指示
                        score = 1.0
                    else:
                        # 一方有有效指示而另一方没有
                        score = 0.0
                    
                    total_score += score
                    valid_pairs += 1
                
                # 6. 计算匹配部分的奖励
                match_reward = total_score / valid_pairs if valid_pairs > 0 else 0.0
                
                # 7. 考虑未匹配部分的惩罚
                # 漏检惩罚：未匹配的真实目标
                miss_penalty = 0.5 * (len(unmatched_gts) / len(gt_items))
                # 误检惩罚：未匹配的预测目标
                extra_penalty = 0.3 * (len(unmatched_preds) / len(gt_items))
                
                # 8. 组合奖励
                reward = max(0.0, min(1.0, match_reward - miss_penalty - extra_penalty))
                rewards.append(reward)
                
            except Exception as e:
                logger.warning(f"计算 indications 奖励时出错: {str(e)}")
                rewards.append(0.0)
                
        return rewards

class CountdownReward(ORM):
    """
    专门评估 countdown_number 属性的奖励函数
    - 处理 None 值
    - 允许 ±1 的误差
    - 对大误差给予部分奖励
    - 处理空列表情况
    """
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
                
                # 3. 匹配框（基于bbox_2d）
                pred_boxes = [item["bbox_2d"] for item in pred_items]
                gt_boxes = [item["bbox_2d"] for item in gt_items]
                matched_pairs, unmatched_gts, unmatched_preds = match_bboxes(
                    gt_boxes, pred_boxes, iou_threshold=0.5
                )
                
                # 4. 初始化奖励
                total_score = 0.0
                valid_pairs = 0
                
                # 5. 对每个匹配对计算 countdown 奖励
                for gt_idx, pred_idx in matched_pairs:
                    pred_item = pred_items[pred_idx]
                    gt_item = gt_items[gt_idx]
                    
                    pred_countdown = pred_item["countdown_number"]
                    gt_countdown = gt_item["countdown_number"]
                    
                    # 计算相似度
                    if gt_countdown is None and pred_countdown is None:
                        score = 1.0
                    elif gt_countdown is not None and pred_countdown is not None:
                        # 数字倒计时 - 允许±1的误差
                        if abs(pred_countdown - gt_countdown) <= 1:
                            score = 1.0
                        else:
                            # 根据误差大小部分奖励
                            error = min(1.0, abs(pred_countdown - gt_countdown) / 10)
                            score = max(0, 0.5 - error)
                    else:
                        # 一方有倒计时另一方没有
                        score = 0.0
                    
                    total_score += score
                    valid_pairs += 1
                
                # 6. 计算匹配部分的奖励
                match_reward = total_score / valid_pairs if valid_pairs > 0 else 0.0
                
                # 7. 考虑未匹配部分的惩罚
                # 漏检惩罚：未匹配的真实目标
                miss_penalty = 0.5 * (len(unmatched_gts) / len(gt_items))
                # 误检惩罚：未匹配的预测目标
                extra_penalty = 0.3 * (len(unmatched_preds) / len(gt_items))
                
                # 8. 组合奖励
                reward = max(0.0, min(1.0, match_reward - miss_penalty - extra_penalty))
                rewards.append(reward)
                
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
orms['external_bbox'] = BboxReward
orms['external_state'] = StateReward
orms['external_indication'] = IndicationsReward
orms['external_countdown'] = CountdownReward