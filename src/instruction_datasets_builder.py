import os
import json
import argparse
import yaml
import glob
from pathlib import Path
from PIL import Image
from qwen_vl_utils import smart_resize
from tqdm import tqdm

# 导入自定义模块
from utils.annotation_processing import translate_annotations_zh_to_en, generate_metainfo, generate_annswer
from utils.visualization import visualize_annotation
from utils.instruction_generator import build_instruction_data, build_instruction_data_end2end, \
    build_instruction_data_noCoT, build_instruction_data_noCoT_end2end
from utils.generate_CoT import generate_CoT_by_template, generate_CoT_by_ai, generate_CoT_by_template_end2end, \
    generate_answer_in_CoT
from utils.config_loader import load_config


"""
本代码旨在创建用于ms-swift框架训练相关的指令微调数据集.
数据集格式遵循: https://swift.readthedocs.io/zh-cn/latest/Customization/%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86.html
数据生成方法由 utils.instruction_generator 提供.
修改其中的代码可以用于其他框架.
本代码对数据的处理方式与交通信号灯标注规范高度相关.
"""

def get_target_size(mode, original_width, original_height):
    """
    根据模式计算目标尺寸
    :param mode: 尺寸调整模式 ('smart', 'shrink', 'original')
    :param original_width: 原始宽度
    :param original_height: 原始高度
    :return: (target_width, target_height)
    """
    if mode == 'smart_resize':
        # 智能调整尺寸：Qwen 官方模式
        new_h, new_w = smart_resize(height=original_height, width=original_width)
        if(original_height > 1940):
            print("超出的原始尺寸",original_height,original_width)
            print("放缩后的",new_h,new_w)
        return new_w, new_h
    elif mode == 'shrink':
        # 固定比例缩放调整尺寸
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
    else:  # 'original'
        # 不调整尺寸
        return original_width, original_height

def main(args, config):
    # 从配置中获取路径
    images_folder = config['data']['images_folder']
    annotations_folder = config['data']['annotations_folder']
    instructions_folder = config['data']['instructions_folder']
    processed_image_folder = config['data']['processed_image_folder']

    os.makedirs(instructions_folder, exist_ok=True)

    # 初始化指令列表
    instructions = []

    # 获取所有标注文件列表
    annotation_files = glob.glob(os.path.join(annotations_folder, '*.json'))
    total_files = len(annotation_files)

    # 初始化图像生成文件夹
    if args.end2end:
        image_output_folder = os.path.join(processed_image_folder, f"{args.resize_mode}")
    else:
        image_output_folder = os.path.join(processed_image_folder, f"marked-{args.resize_mode}-{args.seed}")
    os.makedirs(image_output_folder, exist_ok=True)
    
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
            
        if w > 2500 or h >2500:
            continue 
        
        # 根据模式计算目标尺寸
        target_w, target_h = get_target_size(args.resize_mode, w, h)
        width_ratio = target_w / w
        height_ratio = target_h / h
        
        annotations_en = translate_annotations_zh_to_en(annotations_zh)
        metainfo = generate_metainfo(annotation_en=annotations_en, seed=args.seed, scaling_factor=(width_ratio, height_ratio))

        if args.end2end:
            # 端到端
            if args.cot_mode == "cot_template":
                CoT = generate_CoT_by_template_end2end(metainfo)
            elif args.cot_mode == "noCoT":
                CoT = generate_answer_in_CoT(generate_annswer(metainfo))              
            else:
                print("目前尚未支持该组合参数")
                return
        else:
            # 非端到端
            if args.cot_mode == "cot_template":
                CoT = generate_CoT_by_template(metainfo)
            elif args.cot_mode == "cot_AI":
                try:
                    CoT = generate_CoT_by_ai(metainfo=metainfo, base_name=base_name)
                except:
                    continue
            elif args.cot_mode == "noCoT":
                CoT = generate_answer_in_CoT(generate_annswer(metainfo))
        
        resized_img_path = os.path.join(image_output_folder, f"{base_name}.jpg")
        if args.visualize:
            # 端到端画图不做标记, 只执行图片处理方式。
            if args.end2end:
                visualize_annotation(
                    image_path=img_file,
                    metainfo=metainfo,
                    property_tags=False,
                    show_scene=False,
                    output_image_path=resized_img_path,
                    target_size=(target_w, target_h),
                    label_mode='random_light',
                    marked=False
                )
            else:
                visualize_annotation(
                    image_path=img_file,
                    metainfo=metainfo,
                    property_tags=False,
                    show_scene=False,
                    output_image_path=resized_img_path,
                    target_size=(target_w, target_h),
                    label_mode='random_light',
                    marked=True
                )
        
        # 构建指令数据
        if args.end2end:
            if args.cot_mode == "noCoT":
                entry = build_instruction_data_noCoT_end2end(
                    assistant_content=CoT,
                    image_path=str(resized_img_path)
                )
            else:
                entry = build_instruction_data_end2end(
                    assistant_content=CoT,
                    image_path=str(resized_img_path)
                )
        else:
            if args.cot_mode == "noCoT":
                entry = build_instruction_data_noCoT(
                    bbox_hint=metainfo["bboxes"],
                    assistant_content=CoT,
                    image_path=str(resized_img_path), 
                )
            else:
                entry = build_instruction_data(
                    bbox_hint=metainfo["bboxes"],
                    assistant_content=CoT,
                    image_path=str(resized_img_path)
                )
        instructions.append(entry)
    
    if args.end2end:
        instructions_path = os.path.join(instructions_folder,f"end2end-{args.resize_mode}-{args.cot_mode}-{args.seed}.json")
    else:
        instructions_path = os.path.join(instructions_folder,f"marked-{args.resize_mode}-{args.cot_mode}-{args.seed}.json")
    # 保存指令数据
    with open(instructions_path, 'w', encoding='utf-8') as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
    print(f"指令数据已保存至: {instructions_path}")
    if args.visualize:
        print(f"图片已生成至：{image_output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交通灯数据集构建工具")
    parser.add_argument('--config', type=str, default='/mnt/xiayuxuan/Qwen_TF/Project-2_trace_sft_prompt_v2/config/config.yaml', 
                        help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--resize_mode', type=str, choices=['smart_resize', 'shrink', 'original'], default='smart_resize',
                        help='尺寸调整模式: smart_resize(Qwen官方智能调整), shrink(固定比例缩放), original(不调整、原图)')
    parser.add_argument('--cot_mode', type=str, choices=['cot_template', 'cot_AI', 'noCoT'], default='cot_template', help='CoT数据生成模式: 基于模板、基于AI、无CoT')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    parser.add_argument('--end2end', type=bool, default=False, help='是否端到端（True/False，默认False）')
    parser.add_argument('--visualize', type=bool, default=True, help='是否生成带标注的可视化图像（True/False，默认True）')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # 运行主程序
    main(args, config)