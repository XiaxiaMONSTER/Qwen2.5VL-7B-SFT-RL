import json
import random
import os
from typing import List, Dict

def generate_qa_pairs(
    annotation_path: str, 
    image_folder: str, 
    output_path: str,
    valid_states: List[str] = ["red", "green", "yellow", "off"]
) -> None:
    """
    生成QA对的JSON文件，使用精确的bbox坐标提问
    
    参数:
        annotation_path: 标注文件路径
        image_folder: 裁剪图像文件夹路径
        output_path: 输出JSON文件路径
        valid_states: 有效的灯态列表
    """
    # 读取标注文件
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    qa_pairs = []
    
    for image_name, lights in annotations.items():
        # 过滤掉无效状态
        valid_lights = [
            light for light in lights 
            if light.get("state") in valid_states
        ]
        
        if not valid_lights:
            continue
        
        # 随机选择一个交通灯
        # for selected_light in valid_lights:
        selected_light = random.choice(valid_lights)
        bbox = selected_light["bbox"]
        
        # 计算像素数
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        pixel_count = width * height
        
        # 创建精确的bbox描述
        bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
        
        # 创建QA对
        qa_pairs.append({
            "image_path": os.path.join(image_folder, image_name),
            "question": f"What is the state of the traffic light at {bbox_str} in the picture? Answer with one of the following: green, yellow, red, off.",
            "answer": selected_light["state"],
            "pixel": pixel_count
        })
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"已生成 {len(qa_pairs)} 个QA对，保存至 {output_path}")
    
    # 返回生成的QA对数量
    return len(qa_pairs)

# 使用示例
if __name__ == "__main__":
    # 替换为实际路径
    annotation_path = "/root/autodl-tmp/pure_code/data/json/ROI_448.json"
    image_folder = "/root/autodl-tmp/pure_code/data/processed/ROI_448"
    output_path = "/root/autodl-tmp/pure_code/data/instructions/qa_pairs.json"

    num_qa = generate_qa_pairs(
        annotation_path=annotation_path,
        image_folder=image_folder,
        output_path=output_path
    )
    print(f"成功生成 {num_qa} 个QA对")