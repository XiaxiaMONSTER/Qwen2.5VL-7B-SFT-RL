import json
import os
import glob

from tqdm import tqdm
from PIL import Image

from instruction_datasets_builder import get_target_size
from utils.visualization import visualize_annotation, visual_model_response
from utils.instruction_generator import build_instruction_data
from utils.generate_CoT import generate_answer_in_CoT, generate_CoT_by_template

def extract_metainfo(data, w, h):
    """
    处理交通灯数据字典，转换格式并提取关键信息
    
    参数:
    data: dict - 包含交通灯检测结果的原始字典
    
    返回:
    dict - 转换后的metainfo格式字典
    """
    # 获取交通灯总数
    num_lights = len(data["detections"])
    
    # 初始化结果字典
    metainfo = {
        "scene": None,
        "total_num_of_traffic_lights": num_lights,
        "focusedTrafficLightIndex": [],
        "bboxes": [],
        "type": ["unknown"] * num_lights,
        "pictogram": [],
        "direction": [],
        "light_state": [],
        "indication": [],
        "attachment": ["unknown"] * num_lights
    }
    
    # 遍历所有检测结果
    for i, detection in enumerate(data["detections"]):
        # 坐标转换：ccwh百分比 -> xyxy百分比
        center_x, center_y, width, height = detection["box"]
        x_min = int((center_x - width / 2)*w + 0.5)
        y_min = int((center_y - height / 2)*h + 0.5)
        x_max = int((center_x + width / 2 )*w + 0.5)
        y_max = int((center_y + height / 2)*h + 0.5)
        
        # 添加到bboxes列表
        metainfo["bboxes"].append([x_min, y_min, x_max, y_max])
        
        # 检查是否为focused交通灯
        if "relevant traffic light" in detection["labels"]:
            metainfo["focusedTrafficLightIndex"].append(i)
        
        # 添加其他属性
        metainfo["pictogram"].append(detection["pictogram"])
        metainfo["direction"].append(detection["direction"])
        metainfo["light_state"].append(detection["state"])
        metainfo["indication"].append(detection["indication"])
    
    return metainfo

def main():
    images_folder = r"/root/autodl-tmp/traffic_lights/new_task/images"
    ann_folder = r"/root/autodl-tmp/traffic_lights/new_task/ann"

    processed_image_folder = r"/root/autodl-tmp/traffic_lights/data/processed/new_tasks_images"
    os.makedirs(processed_image_folder, exist_ok=True)

    instructions = []

    annotation_files = glob.glob(os.path.join(ann_folder, '*.json'))
    total_files = len(annotation_files)

    print(f"开始处理 {total_files} 个标注文件...")
    for ann_file in tqdm(annotation_files, desc="处理标注文件", unit="文件", total=total_files):
        base_name = os.path.splitext(os.path.basename(ann_file))[0]
        img_file = os.path.join(images_folder, f"{base_name}.jpg")
        resized_img_path = os.path.join(processed_image_folder, f"{base_name}.jpg")

        # 检查图片是否存在
        if not os.path.exists(img_file):
            print(f"警告：找不到对应的图片文件 {img_file}，跳过")
            continue
        
        # 读取标注数据
        with open(ann_file, 'r', encoding='utf-8') as f:
            ann = json.load(f)
        if len(ann["detections"]) == 0:
            print("无红绿灯标注，跳过")
            continue
        # 打开图片获取原始尺寸
        with Image.open(img_file) as img:
            w, h = img.size
        
        # 根据模式计算目标尺寸
        target_w, target_h = get_target_size("smart_resize", w, h)
        width_ratio = target_w / w
        height_ratio = target_h / h

        metainfo = extract_metainfo(ann, target_w, target_h)

        visualize_annotation(
            image_path=img_file,
            metainfo=metainfo,
            property_tags=False,
            show_scene=False,
            output_image_path=resized_img_path,
            target_size=(target_w, target_h),
            label_mode='random_light'
        )
        
        entry = build_instruction_data(
            bbox_hint=metainfo["bboxes"],
            assistant_content=generate_CoT_by_template(metainfo),
            image_path=str(resized_img_path)
        )

        processed_data = entry.copy()
        # 创建一个新的消息列表，仅保留非assistant消息
        new_messages = []
        for message in processed_data["messages"]:
            if message["role"] != "assistant":
                new_messages.append(message)
        
        # 更新处理后的数据
        processed_data["messages"] = new_messages
        instructions.append(processed_data)

    # config = load_config()
    # instructions_folder = config["data"]["instructions_folder"]
    # instructions_path = os.path.join(instructions_folder,f"marked-new_task.json")

    instructions_path = r"/root/autodl-tmp/traffic_lights/new_task/new_task.json"

    with open(instructions_path, 'w', encoding='utf-8') as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
    print(f"指令数据已保存至: {instructions_path}")


if __name__ == "__main__":
    main()


