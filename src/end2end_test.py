import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
def draw_bounding_boxes(image_path, bboxes, output_path="output.jpg", 
                        box_color=(255, 0, 0), box_thickness=2, 
                        label_color=(255, 255, 255), font_size=15):
    """
    在图像上绘制边界框并保存结果
    
    参数:
    image_path: 输入图像路径
    bboxes: 边界框列表，格式为 [[x_min, y_min, x_max, y_max], ...]
    output_path: 输出图像保存路径
    box_color: 边界框颜色 (BGR格式)
    box_thickness: 边界框线宽
    label_color: 标签文字颜色
    font_size: 字体大小
    """
    # try:
    # 读取图像
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像，可能是不支持的格式: {image_path}")
    
    # 转换到PIL格式（更好的文本渲染支持）
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    
    # 绘制每个边界框
    for i, bbox in enumerate(bboxes):
        if len(bbox) != 4:
            print(f"警告: 边界框 #{i+1} 格式错误，跳过")
            continue
            
        try:
            x_min, y_min, x_max, y_max = map(int, bbox)
            
            # 验证坐标有效性
            if not (0 <= x_min < x_max <= pil_image.width and 0 <= y_min < y_max <= pil_image.height):
                print(f"警告: 边界框 #{i+1} 坐标无效，跳过")
                continue
            
            # 绘制边界框
            draw.rectangle([x_min, y_min, x_max, y_max], 
                            outline=box_color, 
                            width=box_thickness)
            
            # 添加标签
            label = f"traffic light {i+1}"
            
            # 使用textbbox获取文本尺寸 (Pillow 9.0+兼容)
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 计算标签位置
            label_y = max(0, y_min - text_height - 5)
            
            # 标签背景框
            draw.rectangle([x_min, label_y, 
                            x_min + text_width + 5, label_y + text_height + 5],
                            fill=box_color)
            
            # 标签文本
            draw.text((x_min + 2, label_y + 2), 
                        label, fill=label_color)
            
        except Exception as e:
            print(f"处理边界框 #{i+1} 时出错: {str(e)}")
    
    # 创建输出目录（如果不存在）
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换回OpenCV格式并保存
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存至: {output_path}")
    return result_image
    
    # except Exception as e:
    #     print(f"发生错误: {str(e)}")
    #     return None

if __name__ == "__main__":

    # 处理边界框输入
    image_path = "./data/processed/smart_resize/image_02464.jpg"
    bboxes = [[945, 338, 960, 375],[1098, 338, 1113, 375]]
    
    # 绘制边界框
    draw_bounding_boxes(image_path=image_path, bboxes=bboxes)