import os
import json
import argparse
import yaml
from pathlib import Path
from PIL import Image
from qwen_vl_utils import smart_resize
import glob
from tqdm import tqdm
# 导入自定义模块
from utils.annotation_processing import translate_annotations_zh_to_en, generate_metainfo
from utils.visualization import visualize_annotation
from utils.instruction_generator import build_instruction_data
from utils.generate_CoT import generate_CoT_by_template
from utils.config_loader import load_config

def get_target_size(mode, original_width, original_height):
    """
    根据模式计算目标尺寸
    :param mode: 尺寸调整模式 ('smart', 'fixed_ratio', 'none')
    :param original_width: 原始宽度
    :param original_height: 原始高度
    :return: (target_width, target_height)
    """
    if mode == 'smart_resize':
        # 智能调整尺寸
        new_h, new_w = smart_resize(height=original_height, width=original_width)
        return new_w, new_h
    elif mode == 'fixed_ratio':
        # 固定比例调整尺寸
        scale = original_width / original_height
        if scale >= 1.7:
            return 784, 448
        elif scale >= 1.6:
            return 728, 448
        elif scale >= 1.3:
            return 588, 448
        else:
            # 默认尺寸
            return 448, 448
    else:  # 'none'
        # 不调整尺寸
        return original_width, original_height

def main(args, config):
    # 从配置中获取路径
    images_folder = config['data']['images_folder']
    annotations_folder = config['data']['annotations_folder']
    instructions_folder = config['data']['instructions_folder']
    processed_image_folder = config['data']['processed_image_folder']

    # 创建输出目录
    os.makedirs(processed_image_folder, exist_ok=True)
    
    # 初始化指令列表
    instructions = []
    
    output_folder = os.path.join(processed_image_folder, f"marked-{args.resize_mode}")

    os.makedirs(output_folder, exist_ok=True)

    # 获取所有标注文件列表
    annotation_files = glob.glob(os.path.join(annotations_folder, '*.json'))
    total_files = len(annotation_files)

    print(f"开始处理 {total_files} 个标注文件...")

    # 遍历所有标注文件
    for ann_file in tqdm(annotation_files, desc="处理标注文件", unit="文件", total=total_files):
        base_name = os.path.splitext(os.path.basename(ann_file))[0]
        img_file = os.path.join(images_folder, f"{base_name}.jpg")
        
        # 检查图片是否存在
        if not os.path.exists(img_file):
            print(f"警告：找不到对应的图片文件 {img_file}，跳过")
            continue
        
        # 读取标注数据
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations_zh = json.load(f)
        
        # 打开图片获取原始尺寸
        with Image.open(img_file) as img:
            w, h = img.size
        
        # 根据模式计算目标尺寸
        target_w, target_h = get_target_size(args.resize_mode, w, h)
        width_ratio = target_w / w
        height_ratio = target_h / h
        
        # 处理标注数据
        if args.cot_mode == "cot_template":
            annotations_en = translate_annotations_zh_to_en(annotations_zh)
            metainfo = generate_metainfo(annotation_en=annotations_en, scaling_factor=(width_ratio, height_ratio))
            CoT = generate_CoT_by_template(metainfo)
        
        # 输出的图片地址
        resized_img_path = os.path.join(output_folder, f"{base_name}.jpg")

        visualize_annotation(
            image_path=img_file,
            metainfo=metainfo,
            property_tags=False,
            show_scene=False,
            output_image_path=resized_img_path,
            target_size=(target_w, target_h),
            label_mode='random_light'
        )
        
        # 构建指令数据
        entry = build_instruction_data(
            bbox_hint=metainfo["bboxes"],
            cot=CoT,
            image_path=str(resized_img_path))
        instructions.append(entry)

    instructions_path = os.path.join(instructions_folder,f"marked-{args.resize_mode}-{args.cot_mode}.json")
    # 保存指令数据
    with open(instructions_path, 'w', encoding='utf-8') as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
    print(f"指令数据已保存至: {instructions_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交通灯数据集构建工具")
    parser.add_argument('--config', type=str, default='./config/config.yaml', 
                        help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--resize_mode', type=str, choices=['smart_resize', 'fixed_ratio', 'none'], default='smart_resize',
                        help='尺寸调整模式: smart(智能调整), fixed_ratio(固定比例), none(不调整)')
    parser.add_argument('--cot_mode', type=str, choices=['cot_template', 'cot_AI'], default='cot_template', help='CoT数据生成模式：基于模板、基于AI')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    parser.add_argument('--visualize', action='store_true', help='是否生成带标注的可视化图像')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 运行主程序
    main(args, config)