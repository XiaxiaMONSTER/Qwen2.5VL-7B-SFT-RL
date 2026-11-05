import os
import math
import random
import json
import numpy as np

from utils.config_loader import load_config
from tqdm import tqdm

def translate_annotations_zh_to_en(annotations_zh: dict) -> dict:
    """
    将原始标注文件字典转换为英文
    """
    config = load_config()
    translation_maps = config["translation_maps"]
    def process_value(key, value):
        if key not in translation_maps:
            return value
            
        if isinstance(value, list):
            return [translation_maps[key].get(item, item) for item in value]
        else:
            return translation_maps[key].get(value, value)
    # 递归处理嵌套结构
    def recursive_translate(d):
        if isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                new_v = process_value(k, v)
                if isinstance(new_v, (dict, list)):
                    new_dict[k] = recursive_translate(new_v)
                else:
                    new_dict[k] = new_v
            return new_dict
        elif isinstance(d, list):
            return [recursive_translate(item) for item in d]
        else:
            return d
        
    # 执行翻译转换
    annotations_en = recursive_translate(annotations_zh)

    # 基于规则的后处理
    # 对于倒计时灯，属性"relevance"修改为不相关
    for light in annotations_en["traffic_light"]:
        if light["type"] == "countdown lights":
            light["relevance"] = "irrelevant"

    del annotations_en['owner']
    del annotations_en['annotator']

    return annotations_en

def generate_metainfo(annotation_en: dict, seed=42, scaling_factor=(1.0, 1.0)) -> dict:
    """
    生成METAINFO字典
    Args:
        annotation_en: 处理后的英文标注结果
        seed: 随机种子, 默认42
        scaling_factor: 缩放因子, 用于处理图片bbox相关属性. 会改变bbox的标注信息. 
    Returns:
        METAINFO字典
        focusedTrafficLightIndex 是关注灯的序号,从1开始. 
    """
    # 提取场景信息
    scene = annotation_en["scene"]
    
    # 获取交通信号灯列表并打乱顺序
    traffic_lights = annotation_en["traffic_light"].copy()
    random.seed(seed)  # 设置随机种子保证结果可复现
    random.shuffle(traffic_lights)  # 打乱信号灯顺序
    
    # 计算总数
    total_num_of_traffic_lights = len(traffic_lights)
    
    # 初始化存储容器
    focusedTrafficLightIndex  = []
    bboxes, types, pictograms, directions = [], [], [], []
    light_states, indications, attachments = [], [], []
    
    # 遍历处理每个信号灯
    for idx, light in enumerate(traffic_lights):
        # 记录relevant信号灯编号
        if scene == "passing the intersection":
            if light["type"] == "traffic lights for motor vehicles at intersections" and light["direction"] == "front":
                focusedTrafficLightIndex.append(idx + 1)
        elif light["relevance"] == "relevant":
            focusedTrafficLightIndex.append(idx + 1)
        x1, y1, x2, y2 = light["bbox"]
        width_ratio, height_ratio = scaling_factor
        bboxes.append([
            round(x1 * width_ratio),
            round(y1 * height_ratio),
            round(x2 * width_ratio),
            round(y2 * height_ratio)
        ])
        types.append(light["type"])
        pictograms.append(light["pictogram"])
        directions.append(light["direction"])
        light_states.append(light["state"])
        indications.append(light["indication"])
        attachments.append(light["attachment"])
    
    # 构建结果字典
    result = {
        "scene": scene,
        "total_num_of_traffic_lights": total_num_of_traffic_lights,
        "focusedTrafficLightIndex": focusedTrafficLightIndex,
        "bboxes": bboxes,
        "type": types,
        "pictogram": pictograms,
        "direction": directions,
        "light_state": light_states,
        "indication": indications,
        "attachment": attachments
    }
    return result

def countdown_binding(metainfo: dict) -> dict:
    """
    输出形式为 {"1":[4,3]} 表示4和3都绑定1号倒计时灯.
    {"1":[1]}, 表示1绑定自身.
    """
    result = {}
    # 找出所有包含 "countdown" 的 pictogram 的索引（从0开始）
    countdown_indices = [i for i, p in enumerate(metainfo["pictogram"]) if "countdown" in p]

    if not countdown_indices:
        return {}
    
    # 非倒计时灯索引
    non_countdown_indices = set(range(metainfo["total_num_of_traffic_lights"])) - set(countdown_indices)

    for countdown_idx in countdown_indices:
        countdown_number = countdown_idx + 1  # 序号从1开始
        pictogram = metainfo["pictogram"][countdown_idx]
        
        # 检查是否自带图形：圆形或箭头（严格匹配）
        if any("circle" in p or p.endswith("arrow") for p in pictogram):
            result[countdown_number] = [countdown_number]  # 绑定自身
            continue

        # 纯数字倒计时灯：寻找绑定目标
        matches = []
        c_state = metainfo["light_state"][countdown_idx]
        c_direction = metainfo["direction"][countdown_idx]
        c_attachment = metainfo["attachment"][countdown_idx]
        
        # 使用中心点坐标计算距离
        cx1, cy1, cx2, cy2 = metainfo["bboxes"][countdown_idx]
        c_center = [(cx1 + cx2) / 2, (cy1 + cy2) / 2]
        
        for i in non_countdown_indices:  # 只遍历普通灯
            # 检查灯态、方向、挂件一致性
            if (
                metainfo["light_state"][i] == c_state and
                metainfo["direction"][i] == c_direction and
                metainfo["attachment"][i] == c_attachment
            ):
                # 计算中心点距离
                x1, y1, x2, y2 = metainfo["bboxes"][i]
                t_center = [(x1 + x2) / 2, (y1 + y2) / 2]
                distance = math.dist(c_center, t_center)
                matches.append((i + 1, distance))  # (序号, 距离)

        # 按距离排序并提取序号
        matches.sort(key=lambda x: x[1])
        result[countdown_number] = [idx for idx, _ in matches]

    return result


    for countdown_idx in countdown_indices:
        # 获取当前 countdown 灯的信息
        countdown_light_state = metainfo["light_state"][countdown_idx]
        countdown_direction = metainfo["direction"][countdown_idx]
        countdown_attachment = metainfo["attachment"][countdown_idx]
        countdown_bbox = metainfo["bboxes"][countdown_idx][:2]  # 左上角 (x1, y1)
        countdown_number = countdown_idx + 1  # 序号从1开始
        countdown_pictogram = metainfo["pictogram"][countdown_idx]
        matches = []
        # 从自己开始检查: 如果倒计时在自身则自我绑定
        if 'circle' in countdown_pictogram or any(x.endswith('arrow') for x in countdown_pictogram):
            result[str(countdown_number)] = [countdown_number]
        else:
            for i in range(metainfo["total_num_of_traffic_lights"]):
                if i == countdown_idx:
                    continue  # 跳过自己
                light_state = metainfo["light_state"][i]
                direction = metainfo["direction"][i]
                attachment = metainfo["attachment"][i]
                bbox = metainfo["bboxes"][i][:2]
                # 判断是否满足匹配条件
                if (
                    light_state == countdown_light_state and
                    direction == countdown_direction and
                    attachment == countdown_attachment
                ):
                    dx = bbox[0] - countdown_bbox[0]
                    dy = bbox[1] - countdown_bbox[1]
                    distance = math.hypot(dx, dy)
                    matches.append((i + 1, distance))  # 保存序号和距离

            # 按距离升序排序，并去掉距离值，只保留序号
            matches.sort(key=lambda x: x[1])
            matched_indices = [idx for idx, _ in matches]

            # 添加到结果字典中
            result[str(countdown_number)] = matched_indices

    return result

def generate_annswer(metainfo: dict) -> dict:
    scene = metainfo["scene"]
    answer = dict(scene=scene, traffic_lights=[])
    countdown_dict = countdown_binding(metainfo)
    for idx in range(metainfo["total_num_of_traffic_lights"]):
        if int(idx+1) in metainfo["focusedTrafficLightIndex"]:
            countdown_number = None
            if countdown_dict:
                for key,values in countdown_dict.items():
                    if int(idx+1) in values:
                        if "NaN" in metainfo["pictogram"][int(key)-1]:
                            countdown_number = None
                        else:
                            countdown_number = next(int(x) for x in metainfo["pictogram"][int(key)-1] if x.isdigit())
            else:
                countdown_number = None
            answer["traffic_lights"].append({
                "bbox_2d":metainfo["bboxes"][idx],
                "indications":metainfo["indication"][idx],
                "state":metainfo["light_state"][idx],
                "countdown_number":countdown_number
            })
    return answer

## 用来评估metainfo质量
def calculate_annotation_quality(metainfo):
    """计算标注质量分数（0-100）"""
    # 1. 计算bbox大小分数（40分）
    areas = []
    for bbox in metainfo.get('bboxes', []):
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            if width > 0 and height > 0:
                areas.append(width * height)
    
    area_score = 0
    if areas:
        min_area = min(areas)
        avg_area = sum(areas) / len(areas)
        
        # 面积评分：最小面积和平均面积各占20分
        min_area_score = min(20, min_area / 15)  # 最小面积300px²得20分
        avg_area_score = min(20, avg_area / 30)  # 平均面积600px²得20分
        area_score = min_area_score + avg_area_score
    
    # 2. 计算标注完整性分数（30分）
    unknown_count = 0
    total_fields = 0
    
    # 检查pictogram字段
    for picto in metainfo.get('pictogram', []):
        if isinstance(picto, list):
            for item in picto:
                total_fields += 1
                if item in ['unknown', 'NA', 'na', None]:
                    unknown_count += 1
    
    # 检查light_state字段
    for state in metainfo.get('light_state', []):
        total_fields += 1
        if state == 'unknown':
            unknown_count += 1
    
    # 检查indication字段
    for ind in metainfo.get('indication', []):
        total_fields += 1
        if 'unknown' in ind:
            unknown_count += 1
    
    completeness_score = 0
    if total_fields > 0:
        unknown_ratio = unknown_count / total_fields
        completeness_score = 30 * (1 - min(1.0, unknown_ratio * 3))  # 未知率>33%得0分
    
    # 3. 计算bbox数量分数（30分）
    bbox_count = metainfo.get('total_num_of_traffic_lights', len(metainfo.get('bboxes', [])))
    count_score = max(0, 30 - max(0, bbox_count - 10) * 3)  # 超过10个每多1个扣3分
    
    total_score = area_score + completeness_score + count_score
    return round(total_score, 1)

def extract_countdown_info(metainfo):
    """提取倒计时相关信息"""
    has_countdown = False
    countdown_values = []
    
    for picto in metainfo.get('pictogram', []):
        if isinstance(picto, list) and len(picto) > 0:
            # 检查是否包含"countdown"
            if 'countdown' in picto[0].lower():
                has_countdown = True
            
            # 提取所有数字字符串
            for item in picto:
                if isinstance(item, str) and item.isdigit():
                    countdown_values.append(item)
    
    return has_countdown, countdown_values

def extract_bbox_info(metainfo):
    """计算bbox相关指标"""
    areas = []
    for bbox in metainfo.get('bboxes', []):
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            if width > 0 and height > 0:
                areas.append(width * height)
    
    if not areas:
        return 0, 0, 0
    
    return len(metainfo.get('bboxes', [])), min(areas), sum(areas) / len(areas)

def select_annotations(input_folder, output_high, output_low, target_count=3000):
    """
    从文件夹中选择高质量和低质量标注
    
    参数:
    input_folder: 包含标注JSON文件的文件夹路径
    output_high: 高质量标注输出文件
    output_low: 低质量标注输出文件
    target_count: 每种质量的目标数量
    """
    all_scores = []
    file_details = {}
    
    # 收集所有JSON文件
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    print(f"开始评估 {len(json_files)} 个标注文件...")
    
    # 第一阶段：评估所有文件并收集分数
    for filename in tqdm(json_files, desc="评估标注质量"):
        filepath = os.path.join(input_folder, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            annotations_en = translate_annotations_zh_to_en(annotations_zh=data)
            metainfo = generate_metainfo(annotation_en=annotations_en)
            # 计算质量分数
            quality_score = calculate_annotation_quality(metainfo)
            
            # 提取bbox信息
            bbox_count, min_area, avg_area = extract_bbox_info(metainfo)
            
            # 提取倒计时信息
            has_countdown, countdown_values = extract_countdown_info(metainfo)
            
            # 记录文件详情
            file_details[filename] = {
                "score": quality_score,
                "bbox_count": bbox_count,
                "min_bbox_area": min_area,
                "avg_bbox_area": avg_area,
                "has_countdown": has_countdown,
                "countdown_values": countdown_values
            }
            
            all_scores.append(quality_score)
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            continue
    
    # 计算质量分数阈值
    sorted_scores = sorted(all_scores)
    
    # 高质量阈值：前30% (分数最高的30%)
    high_threshold = np.percentile(sorted_scores, 70)
    
    # 低质量阈值：后30% (分数最低的30%)
    low_threshold = np.percentile(sorted_scores, 30)
    
    print(f"\n质量分数分布:")
    print(f"- 最高分: {max(all_scores)}")
    print(f"- 最低分: {min(all_scores)}")
    print(f"- 平均分: {np.mean(all_scores):.1f}")
    print(f"- 高质量阈值: >={high_threshold:.1f}")
    print(f"- 低质量阈值: <={low_threshold:.1f}")
    
    # 第二阶段：选择高质量和低质量文件
    high_quality = {}
    low_quality = {}
    
    for filename, details in file_details.items():
        if details["score"] >= high_threshold and len(high_quality) < target_count:
            high_quality[filename] = details
        elif details["score"] <= low_threshold and len(low_quality) < target_count:
            low_quality[filename] = details
            
        # 如果已经收集足够数量，提前停止
        if len(high_quality) >= target_count and len(low_quality) >= target_count:
            break
    
    # 保存结果
    with open(output_high, 'w') as f_high:
        json.dump({
            "total_files": len(json_files),
            "high_quality_count": len(high_quality),
            "high_threshold": high_threshold,
            "details": high_quality
        }, f_high, indent=2)
    
    with open(output_low, 'w') as f_low:
        json.dump({
            "total_files": len(json_files),
            "low_quality_count": len(low_quality),
            "low_threshold": low_threshold,
            "details": low_quality
        }, f_low, indent=2)
    
    print(f"\n选择完成!")
    print(f"- 高质量标注: {len(high_quality)} 个 (已保存到 {output_high})")
    print(f"- 低质量标注: {len(low_quality)} 个 (已保存到 {output_low})")
    
    return len(high_quality), len(low_quality)