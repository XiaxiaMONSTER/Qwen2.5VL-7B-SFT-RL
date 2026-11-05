import json
import os
import numpy as np
from collections import defaultdict

def analyze_training_convergence(jsonl_file_path, logging_steps, eval_steps, min_epochs=5):
    """
    分析训练日志，判断训练是否收敛及收敛发生的步骤
    
    参数:
    jsonl_file_path: JSONL日志文件路径
    logging_steps: 训练日志记录步长间隔
    eval_steps: 验证集评估步长间隔
    min_epochs: 最小训练轮数（用于避免早期停止判断）
    
    返回:
    dict: 包含收敛判断、收敛步骤和详细分析结果的字典
    """
    # 结果容器
    result = {
        "converged": False,
        "convergence_step": None,
        "convergence_reason": "",
        "reasons": [],
        "metrics": defaultdict(dict),
        "progress": {
            "current_step": 0,
            "max_steps": 0,
            "percentage": 0.0
        },
        "trend_analysis": {}
    }
    
    # 读取并解析日志文件
    try:
        with open(jsonl_file_path, 'r') as f:
            logs = []
            for line in f:
                try:
                    log_data = json.loads(line.strip())
                    logs.append(log_data)
                except json.JSONDecodeError:
                    continue
        
        if not logs:
            result["reasons"].append("日志文件为空")
            return result
    except Exception as e:
        result["reasons"].append(f"读取日志文件失败: {str(e)}")
        return result
    
    # 分离训练日志和验证日志
    train_logs = []
    eval_logs = []
    
    for log in logs:
        # 检查是否是验证日志
        if any(key.startswith("eval_") for key in log.keys()):
            eval_logs.append(log)
        else:
            train_logs.append(log)
    
    # 检查是否有足够的日志数据
    if len(train_logs) < min_epochs * 2:
        result["reasons"].append(f"日志数据不足（仅{len(train_logs)}条训练记录），需要更多数据进行分析")
        return result
    
    # 解析训练进度
    last_log = logs[-1]
    if "global_step/max_steps" in last_log:
        try:
            if isinstance(last_log["global_step/max_steps"], str):
                current, total = last_log["global_step/max_steps"].strip('"').split('/')
            else:
                current = last_log["global_step/max_steps"].get("current", 0)
                total = last_log["global_step/max_steps"].get("max", 1)
            
            current_step = int(current)
            max_steps = int(total)
            progress = current_step / max(1, max_steps)
            
            result["progress"]["current_step"] = current_step
            result["progress"]["max_steps"] = max_steps
            result["progress"]["percentage"] = progress
        except Exception:
            pass
    
    # 提取关键指标
    def extract_metrics(logs, prefix=""):
        metrics = {
            "steps": [],
            "rewards": [],
            "losses": [],
            "kl_divergences": [],
            "clip_ratios": [],
            "format_rewards": [],
            "bbox_rewards": [],
            "state_rewards": [],
            "indications_rewards": [],
            "countdown_rewards": []
        }
        
        for log in logs:
            try:
                # 解析步数
                if "global_step/max_steps" in log:
                    if isinstance(log["global_step/max_steps"], str):
                        current, _ = log["global_step/max_steps"].strip('"').split('/')
                    else:
                        current = log["global_step/max_steps"].get("current", 0)
                    step = int(current)
                    metrics["steps"].append(step)
                
                # 提取指标
                metrics["rewards"].append(log.get(f"{prefix}reward", None))
                metrics["losses"].append(log.get(f"{prefix}loss", None))
                metrics["kl_divergences"].append(log.get(f"{prefix}kl", None))
                metrics["clip_ratios"].append(log.get(f"{prefix}clip_ratio/low_mean", None))
                metrics["format_rewards"].append(log.get(f"{prefix}rewards/FormatReward/mean", None))
                metrics["bbox_rewards"].append(log.get(f"{prefix}rewards/BboxReward/mean", None))
                metrics["state_rewards"].append(log.get(f"{prefix}rewards/StateReward/mean", None))
                metrics["indications_rewards"].append(log.get(f"{prefix}rewards/IndicationsReward/mean", None))
                metrics["countdown_rewards"].append(log.get(f"{prefix}rewards/CountdownReward/mean", None))
            except Exception:
                continue
        
        # 过滤无效值并确保步数一致
        valid_indices = []
        for i in range(len(metrics["steps"])):
            valid = True
            for key in metrics:
                if key != "steps" and metrics[key][i] is None:
                    valid = False
                    break
            if valid:
                valid_indices.append(i)
        
        # 只保留有效数据点
        for key in metrics:
            metrics[key] = [metrics[key][i] for i in valid_indices]
        
        return metrics
    
    # 获取训练和验证指标
    train_metrics = extract_metrics(train_logs)
    eval_metrics = extract_metrics(eval_logs, "eval_")
    
    # 保存指标数据
    result["metrics"]["train"] = train_metrics
    result["metrics"]["eval"] = eval_metrics
    
    # 收敛性分析
    convergence_points = []
    
    # 1. 训练进度检查
    progress = result["progress"]["percentage"]
    if progress >= 0.95:
        result["converged"] = True
        reason = f"训练进度完成: {progress*100:.2f}% >= 95%"
        result["reasons"].append(reason)
        convergence_points.append({
            "step": result["progress"]["current_step"],
            "reason": reason,
            "type": "progress"
        })
    
    # 2. 奖励稳定性分析（训练集和验证集）
    def analyze_reward_stability(metrics, name, steps):
        stability_points = []
        if len(metrics["rewards"]) < 10:
            return stability_points
        
        # 计算滑动窗口稳定性
        for i in range(10, len(metrics["rewards"])):
            rewards = metrics["rewards"][i-9:i+1]  # 最近10个点
            changes = [abs(rewards[j] - rewards[j-1]) for j in range(1, len(rewards))]
            avg_change = sum(changes) / len(changes) if changes else 0
            std_dev = np.std(rewards)
            
            if avg_change < 0.005 and std_dev < 0.01:
                stability_points.append({
                    "step": steps[i],
                    "reason": f"{name}奖励已稳定 (平均变化={avg_change:.5f}, 标准差={std_dev:.5f})",
                    "type": "reward_stability"
                })
        
        return stability_points
    
    # 训练集奖励稳定性
    train_stability_points = analyze_reward_stability(train_metrics, "训练集", train_metrics["steps"])
    # 验证集奖励稳定性
    eval_stability_points = analyze_reward_stability(eval_metrics, "验证集", eval_metrics["steps"])
    
    # 添加到收敛点列表
    convergence_points.extend(train_stability_points)
    convergence_points.extend(eval_stability_points)
    
    # 3. KL散度分析（策略更新程度）
    def analyze_kl_divergence(metrics, name, steps):
        kl_points = []
        if len(metrics["kl_divergences"]) < 5:
            return kl_points
        
        # 检查KL值是否持续很低
        for i in range(4, len(metrics["kl_divergences"])):
            last_kls = metrics["kl_divergences"][i-4:i+1]  # 最近5个点
            if all(kl < 1e-6 for kl in last_kls):
                kl_points.append({
                    "step": steps[i],
                    "reason": f"{name}KL散度持续很低 (最近5个值均 < 1e-6)",
                    "type": "kl_divergence"
                })
        
        return kl_points
    
    train_kl_points = analyze_kl_divergence(train_metrics, "训练集", train_metrics["steps"])
    eval_kl_points = analyze_kl_divergence(eval_metrics, "验证集", eval_metrics["steps"])
    
    convergence_points.extend(train_kl_points)
    convergence_points.extend(eval_kl_points)
    
    # 4. 格式奖励分析（特殊指标）
    def analyze_format_reward(metrics, name, steps):
        format_points = []
        if not metrics["format_rewards"]:
            return format_points
        
        # 检查格式奖励是否达到并保持最大值
        consecutive_max = 0
        for i, value in enumerate(metrics["format_rewards"]):
            if value == 1.0:
                consecutive_max += 1
                if consecutive_max >= 3:  # 连续3次达到最大值
                    format_points.append({
                        "step": steps[i],
                        "reason": f"{name}格式奖励连续3次达到最大值 (1.0)",
                        "type": "format_reward"
                    })
                    consecutive_max = 0  # 重置计数器
            else:
                consecutive_max = 0
        
        return format_points
    
    train_format_points = analyze_format_reward(train_metrics, "训练集", train_metrics["steps"])
    eval_format_points = analyze_format_reward(eval_metrics, "验证集", eval_metrics["steps"])
    
    convergence_points.extend(train_format_points)
    convergence_points.extend(eval_format_points)
    
    # 5. 损失分析
    def analyze_loss(metrics, name, steps):
        loss_points = []
        if len(metrics["losses"]) < 10:
            return loss_points
        
        # 检查损失是否降至极低水平
        for i in range(9, len(metrics["losses"])):
            losses = metrics["losses"][i-9:i+1]  # 最近10个点
            if all(loss < 1e-5 for loss in losses):
                loss_points.append({
                    "step": steps[i],
                    "reason": f"{name}损失持续极低 (最近10个值均 < 1e-5)",
                    "type": "loss"
                })
        
        return loss_points
    
    train_loss_points = analyze_loss(train_metrics, "训练集", train_metrics["steps"])
    eval_loss_points = analyze_loss(eval_metrics, "验证集", eval_metrics["steps"])
    
    convergence_points.extend(train_loss_points)
    convergence_points.extend(eval_loss_points)
    
    # 6. 确定收敛步骤
    if convergence_points:
        # 找到最早的收敛点
        earliest_point = min(convergence_points, key=lambda x: x["step"])
        
        result["converged"] = True
        result["convergence_step"] = earliest_point["step"]
        result["convergence_reason"] = earliest_point["reason"]
        
        # 添加所有收敛原因
        for point in convergence_points:
            if point["step"] == earliest_point["step"]:
                result["reasons"].append(point["reason"])
    
    # 如果没有收敛原因，添加继续训练的建议
    if not result["reasons"]:
        result["reasons"].append("训练尚未收敛，建议继续训练并监控指标")
    
    # 添加详细的收敛点信息
    result["convergence_points"] = sorted(convergence_points, key=lambda x: x["step"])
    
    return result

# 使用示例
if __name__ == "__main__":
    # 替换为实际日志文件路径
    jsonl_file = "/root/autodl-tmp/traffic_lights/models/Qwen2.5VL-CoT-Template-GRPO/v0-20250721-093900/logging.jsonl"
    
    # 假设训练参数
    logging_steps = 5  # 每5步记录一次训练日志
    eval_steps = 100   # 每100步进行一次验证
    
    # 分析训练收敛情况
    analysis_result = analyze_training_convergence(jsonl_file, logging_steps, eval_steps)
    
    # 打印分析结果
    print("=" * 60)
    print("训练收敛分析报告")
    print("=" * 60)
    
    if analysis_result["converged"]:
        print(f"\n✅ 训练已收敛 - 首次收敛发生在步骤 {analysis_result['convergence_step']}")
        print(f"📌 收敛原因: {analysis_result['convergence_reason']}")
    else:
        print("\n🚫 训练尚未收敛")
    
    print("\n训练进度:")
    print(f"当前步数: {analysis_result['progress']['current_step']}")
    print(f"总步数: {analysis_result['progress']['max_steps']}")
    print(f"完成百分比: {analysis_result['progress']['percentage']*100:.2f}%")
    
    print("\n收敛原因汇总:")
    for i, reason in enumerate(analysis_result["reasons"], 1):
        print(f"{i}. {reason}")
    
    if analysis_result["converged"] and "convergence_points" in analysis_result:
        print("\n所有收敛点:")
        for point in analysis_result["convergence_points"]:
            print(f"- 步骤 {point['step']}: {point['reason']} ({point['type']})")
    
    print("\n" + "=" * 60)