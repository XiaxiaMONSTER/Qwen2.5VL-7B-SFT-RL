import os
import re
import json
import argparse
import yaml
import pandas as pd
import csv
from pathlib import Path
from PIL import Image
from qwen_vl_utils import smart_resize
import glob
from tqdm import tqdm
# 导入自定义模块
from utils.annotation_processing import translate_annotations_zh_to_en, generate_metainfo, generate_annswer
from utils.visualization import visualize_annotation
from utils.trace_instruction_generator import build_instruction_data, build_instruction_data_end2end, build_instruction_data_noCoT, build_instruction_data_noCoT_end2end
from utils.generate_CoT import generate_CoT_by_ai, generate_CoT_by_template_end2end, generate_answer_in_CoT, generate_CoT_by_template
from utils.config_loader import load_config


def add_traj_info_to_metainfo(base_name,metainfo, trace_folder, search_window=5, delta_seq=22):

    # 从 metainfo 里提取 frame_id 和对应的轨迹文件名
    base_id, frame_part = base_name.split("_frame_")
    frame_id = int(frame_part)*3
    trace_path = os.path.join(trace_folder, f"{base_id}.txt")

    metainfo["traj_path"] = trace_path

    if not os.path.exists(trace_path):
        metainfo["start_seq"] = None
        metainfo["end_seq"] = None
        return

    # 读取轨迹里的所有 seq
    seq_dict = {}
    with open(trace_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = int(row["seq"])
            seq_dict[seq] = row  # 可以后面用

    # 查找与 frame_id 最接近的 seq_x
    seq_candidates = list(seq_dict.keys())
    if frame_id in seq_candidates:
        seq_x = frame_id
    else:
        # 搜索范围：frame_id ± search_window
        nearby = [i for i in range(frame_id - search_window, frame_id + search_window + 1) if i in seq_candidates]
        if not nearby:
            metainfo["start_seq"] = None
            metainfo["end_seq"] = None
            return
        # 找最近的，距离一样取小的
        seq_x = sorted(nearby, key=lambda x: (abs(x - frame_id), x))[0]

    # 查找 start_seq
    candidate_start = seq_x - delta_seq
    if candidate_start in seq_dict:
        start_seq = candidate_start
    else:
        # 向前找最大可用的序列
        found = [i for i in range(seq_x - 1, seq_x - 25, -1) if i in seq_dict]
        if found:
            start_seq = found[-1]
        else:
            start_seq = seq_x

    # 查找 end_seq
    candidate_end = seq_x + delta_seq
    if candidate_end in seq_dict:
        end_seq = candidate_end
    else:
        found = [i for i in range(seq_x + 1, seq_x + 25) if i in seq_dict]
        if found:
            end_seq = found[-1]
        else:
            end_seq = seq_x

    # 写入 metainfo
    metainfo["start_seq"] = start_seq
    metainfo["end_seq"] = end_seq


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

# ===== 帧号解析 & 映射函数 =====
def parse_frame_index(fn: str) -> int:
    m = re.search(r'frame[_-]?(\d+)', fn)
    if not m:
        raise ValueError(f"无法解析 frame index: {fn}")
    return int(m.group(1))

def map_frame_to_seq(frame_index: int) -> int:
    return (frame_index + 1) * 3 - 1

def summarize_segment_row(row: dict) -> str:
    dur = row['duration_s']
    disp = row['displacement']
    start_v, end_v, avg_v = row['start_speed'], row['end_speed'], row['avg_speed']
    max_v, min_v = row['max_speed'], row['min_speed']
    acc = row['mean_acc']
    turn = row['turn_label']
    angle = row['turn_angle_deg']
    heading_rate = row['heading_rate']

    summary = []
    summary.append(f"- Over {dur:.1f} seconds, the vehicle traveled approximately {disp:.1f} meters.")
    summary.append(f"- The speed changed from {start_v:.2f} m/s to {end_v:.2f} m/s "
                   f"(average {avg_v:.2f} m/s, range {min_v:.2f}–{max_v:.2f} m/s).")
    if abs(acc) < 0.2:
        summary.append("- Acceleration remained low, indicating stable motion.")
    elif acc > 0.2:
        summary.append("- The vehicle was generally accelerating.")
    else:
        summary.append("- The vehicle was generally decelerating.")

    if turn == "直行":
        summary.append(f"- The heading remained stable with minimal change (~{angle:.1f}°).")
    else:
        summary.append(f"- The vehicle made a {turn} with a heading change of {angle:.1f}°, "
                       f"at an average rate of {heading_rate:.1f}°/s.")

    return "Trajectory Summary:\n" + "\n".join(summary)


def main(args, config):
    # 从配置中获取路径
    images_folder = config['data']['images_folder']
    annotations_folder = config['data']['annotations_folder']
    instructions_folder = config['data']['instructions_folder']
    processed_image_folder = config['data']['processed_image_folder']
    trace_folder = config['data']['trace_folder']
    trace_seg_folder = config['data']['trace_seg_folder']

    # 创建输出目录
    os.makedirs(processed_image_folder, exist_ok=True)
    os.makedirs(instructions_folder, exist_ok=True)
    
    # 初始化指令列表
    instructions = []
    
    # 获取所有标注文件列表
    annotation_files = glob.glob(os.path.join(annotations_folder, '*.json'))
    total_files = len(annotation_files)

    # 初始化图像生成文件夹
    if args.end2end:
        image_output_folder = os.path.join(processed_image_folder, f"end2end-trace-{args.resize_mode}-{args.cot_mode}")
    else:
        image_output_folder = os.path.join(processed_image_folder, f"marked-trace-{args.resize_mode}-{args.cot_mode}-{args.seed}")
    os.makedirs(image_output_folder, exist_ok=True)
    

    print(f"开始处理 {total_files} 个标注文件...")

    # 遍历所有标注文件
    for ann_file in tqdm(annotation_files, desc="处理标注文件", unit="文件", total=total_files):
        base_name = os.path.splitext(os.path.basename(ann_file))[0]
        trace_base = base_name.split("_frame_")[0]   
        img_file = os.path.join(images_folder, f"{base_name}.jpg")
        trace_seg_file = os.path.join(trace_seg_folder, f"{trace_base}.csv")
        trace_flag = False
        find_trace = False
        summary_trace_text=''
        
        # 检查图片是否存在
        if not os.path.exists(img_file):
            print(f"警告：找不到对应的图片文件 {img_file}，跳过")
            continue

        # 新增：图片尺寸过滤（宽或高超过2500则跳过）
        with Image.open(img_file) as img:
            w, h = img.size
        if w > 2500 or h > 2500:
            print(f"警告：图片 {img_file} 尺寸过大（{w}x{h}），跳过")
            continue

        if not os.path.exists(trace_seg_file):
            print(f"警告：找不到轨迹分段文件 {trace_seg_file}，不处理轨迹啦！")
        else:
            find_trace = True
            # 解析帧号 → seq
            frame_idx = parse_frame_index(base_name)
            seq = map_frame_to_seq(frame_idx)
            # 读取轨迹分段 CSV
            df = pd.read_csv(trace_seg_file)
            # 找到区间
            row = df[(df['seq_start'] <= seq) & (df['seq_end'] >= seq)]
            if row.empty:
                print(f"错误：没有找到 seq {seq} 的区间 ({trace_seg_file})")
            else:
                row = row.iloc[0].to_dict()
                # 生成总结
                summary_trace_text = summarize_segment_row(row)
                trace_flag = True

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
        
        # 适配 bool 类型参数：直接使用 args.end2end（无需额外转换）
        if args.end2end:
            if args.cot_mode == "cot_template":
                annotations_en = translate_annotations_zh_to_en(annotations_zh)
                metainfo = generate_metainfo(annotation_en=annotations_en, seed=args.seed, scaling_factor=(width_ratio, height_ratio))
                CoT = generate_CoT_by_template_end2end(metainfo)
            # 新增：支持noCoT模式的end2end处理
            elif args.cot_mode == "noCoT":
                annotations_en = translate_annotations_zh_to_en(annotations_zh)
                metainfo = generate_metainfo(annotation_en=annotations_en, seed=args.seed, scaling_factor=(width_ratio, height_ratio))
                CoT = generate_answer_in_CoT(generate_annswer(metainfo))
            else:
                print("目前尚未支持该组合参数")
                return
        else:
            # 处理标注数据
            if args.cot_mode == "cot_template":
                annotations_en = translate_annotations_zh_to_en(annotations_zh)
                metainfo = generate_metainfo(annotation_en=annotations_en, seed=args.seed, scaling_factor=(width_ratio, height_ratio))
                add_traj_info_to_metainfo(base_name,metainfo, trace_folder,search_window=5, delta_seq=22)
                if trace_flag and find_trace:
                    CoT = generate_CoT_by_template(metainfo,traceinfo=row)
                else:
                    CoT = generate_CoT_by_template(metainfo)
            elif args.cot_mode == "cot_AI":
                try:
                    annotations_en = translate_annotations_zh_to_en(annotations_zh)
                    metainfo = generate_metainfo(annotation_en=annotations_en, seed=args.seed, scaling_factor=(width_ratio, height_ratio))
                    CoT = generate_CoT_by_ai(metainfo=metainfo, base_name=base_name)
                except:
                    continue
            elif args.cot_mode == "noCoT":
                annotations_en = translate_annotations_zh_to_en(annotations_zh)
                metainfo = generate_metainfo(annotation_en=annotations_en, seed=args.seed, scaling_factor=(width_ratio, height_ratio))
                answer = generate_answer_in_CoT(generate_annswer(metainfo))
        
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
                if args.visualize:
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
                # 新增：使用build_instruction_data_noCoT_end2end函数
                entry = build_instruction_data_noCoT_end2end(
                    assistant_content=CoT,
                    image_path=str(resized_img_path)
                )
            else:
                entry = build_instruction_data_end2end(
                    metainfo=metainfo,
                    cot=CoT,
                    image_path=str(resized_img_path)
                )
        else:
            if args.cot_mode == "noCoT":
                entry = build_instruction_data_noCoT(
                    bbox_hint=metainfo["bboxes"], 
                    image_path=str(resized_img_path), 
                    answer=answer
                )
            else:
                entry = build_instruction_data(
                    bbox_hint=metainfo["bboxes"],
                    cot=CoT,
                    image_path=str(resized_img_path),
                    summary_trace_text=summary_trace_text if trace_flag else None,
                    trace_flag=trace_flag
                )
        instructions.append(entry)

    if args.end2end:
        instructions_path = os.path.join(instructions_folder,f"end2end-trace-{args.resize_mode}-{args.cot_mode}-{args.seed}.json")
    else:
        instructions_path = os.path.join(instructions_folder,f"marked-trace-{args.resize_mode}-{args.cot_mode}-{args.seed}.json")
    # 保存指令数据
    with open(instructions_path, 'w', encoding='utf-8') as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
    print(f"指令数据已保存至: {instructions_path}")
    
    if args.visualize:
        print(f"图片已生成至：{image_output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交通灯数据集构建工具")
    parser.add_argument('--config', type=str, default='/root/autodl-tmp/traffic_lights/config/config.yaml', 
                        help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--resize_mode', type=str, choices=['smart_resize', 'fixed_ratio', 'none'], default='smart_resize',
                        help='尺寸调整模式: smart(智能调整), fixed_ratio(固定比例), none(不调整)')
    # 新增：支持noCoT模式
    parser.add_argument('--cot_mode', type=str, choices=['cot_template', 'cot_AI', 'noCoT'], default='cot_template', 
                        help='CoT数据生成模式: 基于模板、基于AI、无CoT')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    parser.add_argument('--end2end', type=bool, default=False, help='是否端到端（True/False，默认False）')
    parser.add_argument('--visualize', type=bool, default=True, help='是否生成带标注的可视化图像（True/False，默认True）')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 运行主程序
    main(args, config)