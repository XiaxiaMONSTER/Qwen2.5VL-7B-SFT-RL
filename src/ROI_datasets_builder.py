import os
import json
import cv2
import numpy as np
import sys
import argparse

from collections import deque
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from utils.config_loader import load_config
from utils.annotation_processing import translate_annotations_zh_to_en

def process_all_images(
    image_folder: str,
    annotations_folder: str,
    output_image_folder: str,
    output_annotation_path: str,
    seed: int = 42
) -> None:
    """
    批量处理所有交通场景图片，生成448x448裁剪图及合并标注

    参数:
        image_folder: 原始图片文件夹
        annotations_folder: 原始标注文件夹
        output_image_folder: 裁剪图片输出文件夹
        output_annotation_path: 合并标注输出路径
        seed: 随机种子
    """
    os.makedirs(output_image_folder, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    annotation_files = sorted([f for f in os.listdir(annotations_folder) if f.endswith('.json')])
    
    base_name_map = {}
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        base_name_map[base_name] = img_file
    
    # 创建匹配的文件对列表
    matched_pairs = []
    for ann_file in annotation_files:
        base_name = os.path.splitext(ann_file)[0]
        if base_name in base_name_map:
            matched_pairs.append((base_name, base_name_map[base_name], ann_file))
    
    # 使用tqdm创建进度条
    for base_name, img_file, ann_file in tqdm(
        matched_pairs, 
        desc="处理图片", 
        unit="image",
        total=len(matched_pairs)
    ):
        img_path = os.path.join(image_folder, img_file)
        ann_path = os.path.join(annotations_folder, ann_file)
        
        # 保持原有的try-except结构
        try:
            results = crop_and_generate_annotation(
                img_path, 
                ann_path, 
                output_image_folder,
                seed=seed
            )
            
            save_results(output_annotation_path, results)
            
        except Exception as e:
            print(f"处理 {img_path} 时出错: {str(e)}")
    
    print(f"处理完成! 共处理 {len(image_files)} 张图片，输出到 {output_image_folder}")

def crop_and_generate_annotation(
    image_path: str, 
    annotation_path: str, 
    output_dir: str, 
    seed: int = 42,
    max_generation = 1
) -> Dict[str, List[Dict[str, Any]]]:
    """
    处理单张图片及其标注，生成448x448裁剪图片和对应的交通信号灯属性标注
    
    参数:
        image_path: 原始图片路径
        annotation_path: 原始标注JSON文件路径
        output_dir: 输出目录路径
        seed: 随机种子
        max_generation: 每张图片最大生成量
    返回:
        字典: {
            裁剪图片文件名: [{
                "type": 信号灯类型,
                "pictogram": 图标类型,
                "direction": 物理朝向,
                "state": 灯色状态,
                "bbox": [x1, y1, x2, y2]  # 在448x448图像中的坐标
            }, ...]
        }
    """
    np.random.seed(seed)
    
    # 读取原始图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    H, W = image.shape[:2]
    
    # 读取原始标注
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotation_data = json.load(f)
    
    annotation_data = translate_annotations_zh_to_en(annotation_data)
    
    # 提取交通灯信息
    traffic_lights = annotation_data.get("traffic_light", [])
    if not traffic_lights:
        return {}
    
    # 准备元信息
    bboxes = []
    types = []
    pictograms = []
    directions = []
    states = []
    
    for light in traffic_lights:
        # 转换bbox为整数坐标
        bbox = [int(round(coord)) for coord in light["bbox"]]
        bboxes.append(bbox)
        types.append(light["type"])
        
        # 处理pictogram字段（可能是列表）
        pictogram = light["pictogram"]
        if isinstance(pictogram, list) and pictogram:
            pictograms.append(pictogram[0])
        else:
            pictograms.append(pictogram)
        
        directions.append(light["direction"])
        states.append(light["state"])
    
    # 移除尺寸过滤，保留所有红绿灯
    valid_indices = list(range(len(bboxes)))
    
    # 分组聚类交通灯
    valid_bboxes = [bboxes[i] for i in valid_indices]
    groups = group_traffic_lights(valid_bboxes, cluster_threshold=224)
    results = {}
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for i, group in enumerate(groups[:max_generation]):

        orig_indices = [valid_indices[idx] for idx in group]
        group_bboxes = [bboxes[idx] for idx in orig_indices]
        
        # 变换索引
        group_ = [i for i in range(len(group))]

        # 计算裁剪区域
        crop_box = calculate_crop_box(group_, group_bboxes, W, H)
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        
        # 执行图片裁剪
        crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # 填充到448x448
        padded_img, pad_top, pad_left = pad_to_size_with_position(crop_img, (448, 448))
        
        # 保存裁剪图片
        crop_name = f"{base_name}_{i+1:02d}.jpg"
        crop_path = os.path.join(output_dir, crop_name)
        cv2.imwrite(crop_path, padded_img)
        
        # 生成该裁剪图的信号灯属性标注
        light_annotations = []
        for idx in orig_indices:
            if is_bbox_inside(bboxes[idx], crop_box):
                x1, y1, x2, y2 = bboxes[idx]
                
                # 计算相对坐标
                rel_x1 = x1 - crop_x1
                rel_y1 = y1 - crop_y1
                rel_x2 = x2 - crop_x1
                rel_y2 = y2 - crop_y1
                
                # 转换为填充后坐标
                abs_x1 = rel_x1 + pad_left
                abs_y1 = rel_y1 + pad_top
                abs_x2 = rel_x2 + pad_left
                abs_y2 = rel_y2 + pad_top
                
                # 确保坐标在有效范围内
                abs_x1 = max(0, min(abs_x1, 447))
                abs_y1 = max(0, min(abs_y1, 447))
                abs_x2 = max(0, min(abs_x2, 447))
                abs_y2 = max(0, min(abs_y2, 447))
                
                # 只保留可见的红绿灯
                if abs_x2 - abs_x1 > 4 and abs_y2 - abs_y1 > 4:
                    light_annotations.append({
                        "type": types[idx],
                        "pictogram": pictograms[idx],
                        "direction": directions[idx],
                        "state": states[idx],
                        "bbox": [int(abs_x1), int(abs_y1), int(abs_x2), int(abs_y2)]
                    })
        
        # 添加到结果
        if light_annotations:
            results[crop_name] = light_annotations
    
    return results

def group_traffic_lights(bboxes: List[List[int]], cluster_threshold: int = 224) -> List[List[int]]:
    centers = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
    
    groups = []
    visited = [False] * len(bboxes)
    
    for i in range(len(centers)):
        if not visited[i]:
            group = []
            queue = deque([i])
            visited[i] = True
            
            while queue:
                idx = queue.popleft()
                group.append(idx)
                
                for j in range(len(centers)):
                    if not visited[j]:
                        dist = np.sqrt((centers[idx][0] - centers[j][0])**2 + 
                                      (centers[idx][1] - centers[j][1])**2)
                        if dist < cluster_threshold:
                            visited[j] = True
                            queue.append(j)
            
            groups.append(group)
    
    groups.sort(key=lambda g: max(
        (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1]) for i in g
    ), reverse=True)
    
    return groups

def calculate_crop_box(group: List[int], bboxes: List[List[int]], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    center_x, center_y = 0, 0
    for idx in group:
        x1, y1, x2, y2 = bboxes[idx-1]
        center_x += (x1 + x2) / 2
        center_y += (y1 + y2) / 2
    center_x /= len(group)
    center_y /= len(group)
    
    x1 = int(max(0, center_x - 224))
    y1 = int(max(0, center_y - 224))
    x2 = int(min(img_width, center_x + 224))
    y2 = int(min(img_height, center_y + 224))
    
    if x2 - x1 < 448:
        if x1 == 0:
            x2 = min(img_width, x1 + 448)
        else:
            x1 = max(0, x2 - 448)
    
    if y2 - y1 < 448:
        if y1 == 0:
            y2 = min(img_height, y1 + 448)
        else:
            y1 = max(0, y2 - 448)
    
    return (x1, y1, x2, y2)

def pad_to_size_with_position(image: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    target_w, target_h = size
    
    if h == target_h and w == target_w:
        return image, 0, 0
    
    padded_img = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    pad_top = (target_h - h) // 2
    pad_left = (target_w - w) // 2
    padded_img[pad_top:pad_top+h, pad_left:pad_left+w] = image
    
    return padded_img, pad_top, pad_left

def is_bbox_inside(inner_bbox: List[int], outer_bbox: List[int]) -> bool:
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    return (ix1 >= ox1) and (iy1 >= oy1) and (ix2 <= ox2) and (iy2 <= oy2)

def save_results(output_path: str, new_results: Dict[str, Any]) -> None:
    all_results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        except:
            pass
    
    all_results.update(new_results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    config = load_config()

    parser = argparse.ArgumentParser(
        description='处理交通场景图片，生成448x448裁剪图及合并标注',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加命令行参数
    parser.add_argument('--image_folder', type=str, default=config["data"]["images_folder"],
                        help='原始图片文件夹路径')
    parser.add_argument('--annotations_folder', type=str, default=config["data"]["annotations_folder"],
                        help='原始标注文件夹路径')
    parser.add_argument('--output_image_folder', type=str, default=os.path.join(config["data"]["processed_image_folder"], "ROI_448"),
                        help='裁剪图片输出文件夹路径')
    parser.add_argument('--output_annotation_path', type=str, default='./data/json/ROI_448.json',
                        help='合并标注输出文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.image_folder):
        print(f"错误: 图片文件夹不存在: {args.image_folder}")
        sys.exit(1)
    
    if not os.path.exists(args.annotations_folder):
        print(f"错误: 标注文件夹不存在: {args.annotations_folder}")
        sys.exit(1)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_annotation_path), exist_ok=True)
    os.makedirs(args.output_image_folder, exist_ok=True)
    
    # 执行处理
    process_all_images(
        image_folder=args.image_folder,
        annotations_folder=args.annotations_folder,
        output_image_folder=args.output_image_folder,
        output_annotation_path=args.output_annotation_path,
        seed=args.seed
    )