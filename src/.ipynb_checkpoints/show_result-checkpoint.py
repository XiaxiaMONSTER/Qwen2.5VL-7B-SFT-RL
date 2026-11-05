from utils.visualization import visual_cot, visualize_annotation, visual_model_response
from utils.assessment import match_answer
import json
import os
import argparse
from PIL import Image
from utils.annotation_processing import translate_annotations_zh_to_en, generate_metainfo
from utils.config_loader import load_config
from qwen_vl_utils import smart_resize

# def ensure_directory_exists(path: str) -> None:
#     """确保目录存在，如果不存在则创建"""
#     os.makedirs(os.path.dirname(path), exist_ok=True)

def process_single_sample(index: int, model_output: dict, answer_data: dict, config: dict) -> None:
    """处理单个样本的可视化"""
    # try:
        # 获取基础文件名
    base_name = os.path.splitext(os.path.basename(model_output[index]["image_path"][0]))[0]
    
    # 加载标注数据
    ann_path = os.path.join(config["data"]["annotations_folder"], f"{base_name}.json")
    # ann_path = os.path.join("./datasets/t0715_mp4cz2_irrc2a4_e", f"{base_name}.json")
    with open(ann_path, 'r', encoding='utf-8') as f:
        ann = json.load(f)
    
    # 加载原始图像
    image_path = os.path.join(config["data"]["images_folder"], f"{base_name}.jpg")
    # image_path = os.path.join("./datasets/tasks_images", f"{base_name}.jpg")
    img = Image.open(image_path)
    w, h = img.size
    
    # 计算缩放比例
    new_h, new_w = smart_resize(height=h, width=w)
    width_ratio = new_w / w
    height_ratio = new_h / h

    # # 处理标注信息
    annotations_en = translate_annotations_zh_to_en(annotations_zh=ann)
    answer_metainfo = generate_metainfo(
        annotation_en=annotations_en,
        scaling_factor=(width_ratio, height_ratio)
    )

    # answer_metainfo = extract_metainfo(ann, w, h)
    if answer_metainfo['total_num_of_traffic_lights'] == 0:
        return
    # 创建输出目录
    output_base = config["data"]["processed_image_folder"]
    os.makedirs(os.path.join(output_base, "val_images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "output_images"), exist_ok=True)
    # 生成答案可视化图像
    val_image_path = os.path.join(output_base, "val_images", f"{base_name}.jpg")
    visualize_annotation(
        image_path=image_path,
        metainfo=answer_metainfo,
        property_tags=True,
        show_scene=False,
        output_image_path=val_image_path,
        target_size=(new_w, new_h),
        label_mode="default"
    )
    # 处理模型响应
    cot = model_output[index]["response"].replace("✅", "√").replace("❌", "×").replace("⚠️", "!")
    answer_dict = match_answer(model_output[index]["response"])
    # print("调试信息，match_answer提取后的明模型输出结果：",answer_dict)
    # 生成模型输出可视化图像
    model_output_path = os.path.join(output_base, "output_images", f"{base_name}.jpg")
    visual_model_response(
        image_path=image_path, 
        answer_dict=answer_dict, 
        output_path=model_output_path,
        answer_metainfo=answer_metainfo,
        target_size=(new_w, new_h)
    )
    
    visual_cot(
        img_path1=val_image_path,
        img_path2=model_output_path,
        cot_text=cot,
        text_region_width=1250,
        bg_color=(255, 255, 255),
        title1="result",
        title2="answer"
    )
    
    print(f"成功处理样本 {index}: {base_name}")
        
    # except Exception as e:
    #     print(f"处理样本 {index} 时出错: {str(e)}")

def main(eval_path: str, output_json_path: str, config):
    """主处理函数"""
    
    # 加载模型预测结果
    with open(output_json_path, 'r') as f:
        model_output = json.load(f)["output"]
    
    # 加载测试集答案
    with open(eval_path, 'r') as f:
        answer_data = json.load(f)
    
    # 处理每个样本
    total_samples = len(answer_data)
    print(f"开始处理 {total_samples} 个样本...")
    
    for index in range(total_samples):
        process_single_sample(index, model_output, answer_data, config)
    
    print(f"处理完成! 中间图片：测试集答案结果 保存在 {os.path.join(config['data']['processed_image_folder'], 'val_images')}")
    print(f"处理完成! 中间图片：模型输出结果 保存在 {os.path.join(config['data']['processed_image_folder'], 'output_images')}")
    print(f"处理完成! 思维链结果图片保存在{config['result']['cot_visualization']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理模型输出并生成可视化结果")
    parser.add_argument("--eval_path", default="/mnt/workspace/xiayuxuan/Qwen_TF/Project-2_trace_sft_prompt_v2/data/instructions/eval_dataset.json", help="测试集JSON文件路径")
    parser.add_argument("--output_json_path", default="/mnt/workspace/xiayuxuan/Qwen_TF/Project-2_trace_sft_prompt_v2/results/model_output/CoTv2.2_sft.json", help="模型输出JSON文件路径")
    parser.add_argument('--config', type=str, default='/mnt/workspace/xiayuxuan/Qwen_TF/Project-2_trace_sft_prompt_v2/config/config.yaml', help='配置文件路径 (默认: config.yaml)')
    
    args = parser.parse_args()
    config = load_config(args.config)
    main(args.eval_path, args.output_json_path, config)