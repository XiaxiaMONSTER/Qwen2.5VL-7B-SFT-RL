import cv2
import os
import random
import textwrap
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import matplotlib.image as mpimg

from utils.config_loader import load_config

# 默认颜色设置
DEFAULT_BG_COLOR = (173, 216, 230)  # RGB格式的浅蓝色
DEFAULT_TEXT_COLOR = (0, 0, 0)      # 黑色

def visualize_annotation(image_path, metainfo, answer_metainfo=None, format=".jpg", 
                        property_tags=False, show_scene=False, output_image_path=False,
                        target_size=None, label_mode='default', number_label=True, resize_mode="smart_resize",
                        marked=True):
    
    # 构建图像路径
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image {image_path}")
    
    # 获取原始尺寸
    orig_height, orig_width = image.shape[:2]
    
    # 设置目标尺寸（如果未提供则使用原始尺寸）
    if target_size is not None:
        target_width, target_height = target_size
    else:
        target_width, target_height = orig_width, orig_height
    
    # 获取标注数据
    bboxes = metainfo["bboxes"]
    answer_lights = metainfo.get("focusedTrafficLightIndex", [])
    
    # 调整图像大小（如果需要）
    if (target_width, target_height) != (orig_width, orig_height):
        if target_width > orig_width or target_height > orig_height:
            interpolation = cv2.INTER_LINEAR  # 放大时使用线性插值
        else:
            interpolation = cv2.INTER_AREA   # 缩小时使用区域插值
            
        image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    
    # 添加场景文本
    if show_scene:
        scene_text = metainfo.get("scene", "Unknown Scene").title()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(scene_text, font, font_scale, thickness)

        # 添加边距
        padding = 10

        # 动态设置矩形区域
        top_left = (5, 5)
        bottom_right = (top_left[0] + text_width + padding * 2, 
                        top_left[1] + text_height + baseline + padding)

        # 绘制白色背景矩形
        cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)

        # 绘制黑色文本
        text_origin = (top_left[0] + padding, top_left[1] + text_height + padding // 2)
        cv2.putText(
            image,
            scene_text,
            text_origin,
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    if marked:
        # 为每个交通灯生成随机的浅色背景
        light_colors = []
        for _ in range(len(bboxes)):
            # 生成随机的浅色 (R, G, B) 值在200-255之间
            color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            light_colors.append(color)
        
        # 绘制标注 
        for idx, bbox in enumerate(bboxes):
            # 检查是否是答案框
            is_answer = (idx + 1) in answer_lights
            
            # 转换坐标
            x1, y1, x2, y2 = map(int, bbox)
            
            # 框体扩大
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width_orig = x2 - x1
            height_orig = y2 - y1

            # 动态调整放大倍数：基于框体面积判断
            box_area = width_orig * height_orig  # 计算框体面积（像素²）

            large_size_threshold = 2000
            small_size_threshold = 200


            # 根据面积动态设置放大倍数
            if box_area < small_size_threshold:
                scale = 2  # 小框：特大放大倍数，提升标注可读性
            elif box_area < large_size_threshold:
                scale = 1.5  # 中框：较大放大倍数，提升标注可读性
            else:
                scale = 1.2  # 大框：较小放大倍数，平衡显示与画面占用

            new_width = width_orig * scale
            new_height = height_orig * scale
            x1 = int(center_x - new_width / 2)
            y1 = int(center_y - new_height / 2)
            x2 = int(center_x + new_width / 2)
            y2 = int(center_y + new_height / 2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(target_width - 1, x2)
            y2 = min(target_height - 1, y2)


            # 根据模式选择边框颜色
            if label_mode == 'random_light':
                border_color = light_colors[idx]  # 使用随机浅色作为边框
            else:
                border_color = (0, 255, 0) if is_answer else (0, 0, 255)  # 默认模式使用红/绿色
            
            # 绘制边框 
            cv2.rectangle(image, (x1, y1), (x2, y2), border_color, 2)
            
            # 绘制数字标签
            if number_label: 
                if resize_mode == "smart_resize":
                    font_scale = 0.8
                    thickness = 2
                elif resize_mode == "fixed_ratio":
                    font_scale = 0.4
                    thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(idx + 1)
                
                # 获取文本尺寸（包括基线）
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                total_height = text_height + baseline - 3
                # 确保标签位置在图像范围内
                # 确保标签不会超出图像顶部
                label_x = max(0, x1)
                label_y = max(total_height, y1)
                # 背景框坐标
                bg_x1 = label_x
                bg_y1 = max(0, label_y - total_height + 1)  # 确保背景不会超出图像顶部
                bg_x2 = min(target_width, label_x + text_width)  # 确保不会超出图像右侧
                bg_y2 = min(target_height, label_y - 1)  # 确保不会超出图像底部
                
                # 使用随机浅色作为背景
                bg_color = light_colors[idx]
                
                # 绘制标签背景
                cv2.rectangle(
                    image,
                    (bg_x1, bg_y1),
                    (bg_x2, bg_y2),
                    bg_color,
                    -1
                )
                # 绘制标签文本（黑色）
                if resize_mode == "smart_resize":
                    cv2.putText(
                        image,
                        text,
                        (label_x+1, label_y-1),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                        cv2.LINE_AA
                    )
                elif resize_mode == "fixed_ratio":
                    cv2.putText(
                        image,
                        text,
                        (label_x, label_y),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                        cv2.LINE_AA
                    )


            # 处理属性标签
            if property_tags and metainfo['total_num_of_traffic_lights']>0:
                # 获取预测属性
                pred_state = str(metainfo.get('light_state', [['Unknown']])[idx][0]).title()
                pred_indication = '_'.join(
                    [str(i[0]).title() for i in metainfo.get('indication', [[['Unknown']]])[idx]]
                )
                
                # 完整的属性
                tf_state = metainfo.get('light_state', [['Unknown']])[idx]
                tl_indication = metainfo.get('indication', [[['Unknown']]])[idx][0]

                if pred_state == 'N':
                    #print("完整的灯态(结果是NA)", tf_state)
                    if tf_state == 'NA':
                        print("完整的属性就是NA，不带空格")
                
                
                ##PRINT（pred_state）出来的灯态有R Y G U N O
                state_first_char = pred_state.upper() if pred_state else 'U'
                color_map = {
                    'R': (0, 0, 255),   # 红色
                    'Y': (0, 185, 255), # 橘黄色
                    'G': (0, 255, 0),   # 绿色
                    'U': (0, 0, 0),     # 黑色
                    'O': (255,150,173), # 浅紫色
                    'N': (203,192,255)  # 粉色
                }     
                state_color = color_map.get(state_first_char, (0, 0, 0))  # 背景色沿用原文本色
                

                # 为了区分unknown指示和uturn
                if tl_indication == 'unknown':
                    indication_key = 'unknown'
                else:
                    indication_key = pred_indication.lower().replace(' ', '') if pred_indication else 'unknown'
                indication_color_map = {
                    's': (0, 165, 255), 'l': (0, 0, 255),
                    'r': (0, 255, 0), 'u': (255, 0, 0), 'unknown': (0, 0, 0)
                }
                indication_color = indication_color_map.get(indication_key, (0, 0, 0))  # 背景色映射-默认白色
                
                
                # # 如果有真实标注，进行比较(颜色以灯态颜色为主，正确与否先注释)
                # if answer_metainfo:
                #     # 查找匹配的真实框索引
                #     answer_idx = None
                #     tolerance = 2.0  # 浮点数比较容差
                    
                #     # 遍历真实框，寻找坐标匹配的框
                #     for i, true_bbox in enumerate(answer_metainfo.get("bboxes", [])):
                #         # 检查坐标是否匹配（考虑浮点数精度）
                #         if (abs(bbox[0] - true_bbox[0]) < tolerance and
                #             abs(bbox[1] - true_bbox[1]) < tolerance and
                #             abs(bbox[2] - true_bbox[2]) < tolerance and
                #             abs(bbox[3] - true_bbox[3]) < tolerance):
                #             answer_idx = i
                #             break
                    
                #     # 获取真实属性（如果找到匹配框）
                #     if answer_idx is not None:
                #         true_state = str(
                #             answer_metainfo.get('light_state', [['Unknown']])[answer_idx][0]
                #         ).title()
                #         true_indication = '_'.join(
                #             [str(i[0]).title() for i in 
                #             answer_metainfo.get('indication', [[['Unknown']]])[answer_idx]]
                #         )
                        
                #         # 比较状态和指示
                #         state_correct = (pred_state.lower().replace(' ', '') 
                #                         == true_state.lower().replace(' ', ''))
                #         indication_correct = (pred_indication.lower().replace(' ', '') 
                #                             == true_indication.lower().replace(' ', ''))
                #         # 更新颜色(颜色以灯态颜色为主，正确与否先注释)
                #         state_color = (0, 255, 0) if state_correct else (0, 0, 255)
                #         indication_color = (0, 255, 0) if indication_correct else (0, 0, 255)

                
                # 计算属性标签位置（放在框下方，确保不重叠）
                prop_y = min(target_height - 30, y2 + 20)
                

                # 字体参数（保持不变）
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                text_color = (255, 255, 255)  # 文本改为白色，与背景形成对比
                padding = 2  # 背景内边距


                # 绘制状态文本
                state_text = pred_state if pred_state else 'U'
                                    # 计算文本尺寸（宽、高、基线）
                state_size, state_baseline = cv2.getTextSize(state_text, font, font_scale, thickness)
                # 计算背景位置
                state_bg_x1 = x1
                state_bg_y1 = prop_y - state_baseline - padding -7
                state_bg_x2 = x1 + state_size[0] + padding * 2
                state_bg_y2 = prop_y + padding
                # 绘制背景矩形
                cv2.rectangle(image, (state_bg_x1, state_bg_y1), (state_bg_x2, state_bg_y2), state_color, -1)
                # 绘制文本（白色）
                cv2.putText(image, state_text, (x1 + padding, prop_y), 
                        font, font_scale, text_color, thickness)
                
                # 绘制指示文本（带背景）
                state_width = state_size[0] + padding * 2  # 状态背景总宽度
                indication_x = x1 + state_width + 4  # 保持原有间距

                # 为了区分unknown指示和uturn
                if tl_indication == 'unknown':
                    indication_text = 'u'
                else:
                    indication_text = pred_indication.upper() if pred_indication else 'u'

                # 计算文本尺寸
                ind_size, ind_baseline = cv2.getTextSize(indication_text, font, font_scale, thickness)
                # 计算背景位置
                ind_bg_x1 = indication_x
                ind_bg_y1 = prop_y - ind_baseline - padding - 7
                ind_bg_x2 = indication_x + ind_size[0] + padding * 2
                ind_bg_y2 = prop_y + padding
                # 绘制背景矩形
                cv2.rectangle(image, (ind_bg_x1, ind_bg_y1), (ind_bg_x2, ind_bg_y2), indication_color, -1)
                # 绘制文本（白色）
                cv2.putText(image, indication_text, (indication_x + padding, prop_y), 
                        font, font_scale, text_color, thickness)
                
                # 处理数字部分
                if 'pictogram' in metainfo and idx < len(metainfo['pictogram']):
                    number = next((int(x) for x in metainfo["pictogram"][idx] if x.isdigit()), 0)
                    if number != 0:
                        number_text = f"num:{number}"
                        # 数字文本位置（在属性下方）
                        number_y = min(target_height - 10, prop_y + 20)
                        cv2.putText(image, number_text, (x1, number_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 保存结果 
    if output_image_path:
        if isinstance(output_image_path, str):
            output_path = output_image_path
        else:
            # 从路径中提取基本文件名
            basename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{basename}_marked{format}"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        return output_path

    return image

def resize_and_pad(image, max_size, bg_color):
    """保持原始尺寸，仅在超过最大尺寸时缩小"""
    h, w = image.shape[:2]
    max_w, max_h = max_size
    
    # 计算缩放比例（仅在需要时缩小）
    ratio = 1.0
    if w > max_w:
        ratio = min(ratio, max_w / w)
    if h > max_h:
        ratio = min(ratio, max_h / h)
    
    # 仅在需要时缩放
    if ratio < 1.0:
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h
        resized = image
    
    # 创建背景画布
    bgr_bg_color = (bg_color[2], bg_color[1], bg_color[0])
    canvas = np.full((max_h, max_w, 3), bgr_bg_color, dtype=np.uint8)
    
    # 居中放置图片
    x = (max_w - new_w) // 2
    y = (max_h - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized
    
    return canvas

def add_title_with_pil(image, title, height=60, bg_color=DEFAULT_BG_COLOR, text_color=DEFAULT_TEXT_COLOR):
    """使用PIL在图像顶部添加标题"""
    # 将OpenCV图像转换为PIL格式
    if image is None or image.size == 0:
        pil_img = Image.new('RGB', (1920, 1200), bg_color)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 创建新的标题图像
    title_img = Image.new('RGB', (pil_img.width, height), bg_color)
    draw = ImageDraw.Draw(title_img)
    
    if title:
        # 获取支持特殊符号的字体
        font = get_supported_font(28)
        
        # 计算文本位置
        text_width = draw.textlength(title, font=font)
        text_x = (title_img.width - text_width) // 2
        text_y = (height - font.size) // 2
        
        # 绘制标题
        draw.text((text_x, text_y), title, font=font, fill=text_color)
    
    # 组合标题和图像
    combined = Image.new('RGB', (pil_img.width, height + pil_img.height), bg_color)
    combined.paste(title_img, (0, 0))
    combined.paste(pil_img, (0, height))
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

def get_supported_font(font_size=24):
    """获取支持特殊符号的字体"""
    config = load_config()
    font_names = config["paths"]["font"]
    
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, font_size)
        except:
            continue
    
    # 所有字体都不可用时使用默认字体
    print("警告：未找到支持特殊符号的字体，使用默认字体")
    return ImageFont.load_default()

def render_text_with_pil(text, width, max_height, font_size=24, 
                         bg_color=DEFAULT_BG_COLOR, text_color=DEFAULT_TEXT_COLOR):
    """
    使用PIL渲染文本区域，支持换行符和自动换行，精确计算宽度避免溢出
    """
    font = get_supported_font(font_size)
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))  # 临时draw对象用于测量文本
    
    # 分割文本行（保留手动换行符）
    paragraphs = text.split('\n')
    lines = []
    
    for para in paragraphs:
        # 处理空段落
        if not para.strip():
            lines.append("")
            continue
        
        # 按单词分割段落
        words = para.split(' ')
        current_line = []
        
        for word in words:
            # 测试当前行添加单词后的宽度
            test_line = ' '.join(current_line + [word])
            text_width = draw.textlength(test_line, font=font)
            
            # 如果超过最大宽度（预留20px边距）
            if text_width > (width - 40):  # 左右各留20px边距
                if current_line:  # 当前行非空则换行
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:  # 单个单词超长，强制加入当前行
                    lines.append(word)
            else:
                current_line.append(word)
        
        # 添加当前段落的最后一行
        if current_line:
            lines.append(' '.join(current_line))
    
    # 计算行高和总高度（行高=字体大小*1.4倍）
    line_height = int(font_size * 1.4)
    total_height = len(lines) * line_height + 40  # 上下各留20px边距
    
    # 限制最大高度
    text_height = min(total_height, max_height)
    text_img = Image.new('RGB', (width, text_height), color=bg_color)
    draw = ImageDraw.Draw(text_img)
    
    # 渲染文本行
    y = 20  # 顶部边距
    for i, line in enumerate(lines):
        if y + line_height > text_height - 20:  # 底部预留20px
            # 添加省略号表示内容被截断
            draw.text((20, y), "... (text truncated)", font=font, fill=text_color)
            break
            
        draw.text((20, y), line, font=font, fill=text_color)
        y += line_height
    
    return cv2.cvtColor(np.array(text_img), cv2.COLOR_RGB2BGR), text_height

def visual_cot(img_path1=None, img_path2=None, img_path3=None, cot_text="", 
              title1="", title2="", title3="", cot_title="CoT Reasoning", 
              main_title="Chain-of-Thought Visualization",
              text_region_width=960,  # 文字区域宽度
              padding=20,  # 图片间距
              title_height=40,  # 标题高度
              bg_color=DEFAULT_BG_COLOR,  # 背景颜色
              text_color=DEFAULT_TEXT_COLOR):  # 文字颜色
    """生成CoT可视化图像，图片以原始尺寸展示"""
    config = load_config()
    base_name = os.path.splitext(os.path.basename(img_path1))[0]
    # 加载原始图片
    images = []
    titles = [title1, title2, title3]
    img_paths = [img_path1, img_path2, img_path3]
    
    # 处理每张图片
    max_img_width = 0
    total_img_height = 0
    bgr_bg_color = (bg_color[2], bg_color[1], bg_color[0])
    
    for i, (img_path, title) in enumerate(zip(img_paths, titles)):
        if img_path and os.path.exists(img_path):
            # 加载原始图片
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]
            
            # 更新最大宽度
            if orig_w > max_img_width:
                max_img_width = orig_w
        else:
            # 创建占位图
            orig_w = 600  # 默认宽度
            orig_h = 400  # 默认高度
            orig_img = np.full((orig_h, orig_w, 3), bgr_bg_color, dtype=np.uint8)
            
            # 添加占位符文本
            if title:
                placeholder_pil = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(placeholder_pil)
                font = get_supported_font(28)
                text = title if title else f"Image {i+1} Missing"
                text_width_pil = draw.textlength(text, font=font)
                text_x = (orig_w - text_width_pil) // 2
                text_y = (orig_h - font.size) // 2
                draw.text((text_x, text_y), text, font=font, fill=text_color)
                orig_img = cv2.cvtColor(np.array(placeholder_pil), cv2.COLOR_RGB2BGR)
        
        # 添加标题区域
        title_bar = np.full((title_height, orig_w, 3), bgr_bg_color, dtype=np.uint8)
        title_bar_pil = Image.fromarray(cv2.cvtColor(title_bar, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(title_bar_pil)
        
        if title:
            font = get_supported_font(28)
            text_width_pil = draw.textlength(title, font=font)
            text_x = (orig_w - text_width_pil) // 2
            text_y = (title_height - font.size) // 2
            draw.text((text_x, text_y), title, font=font, fill=text_color)
        
        title_bar = cv2.cvtColor(np.array(title_bar_pil), cv2.COLOR_RGB2BGR)
        
        # 组合图片和标题
        combined_img = np.zeros((orig_h + title_height, orig_w, 3), dtype=np.uint8)
        combined_img[:title_height, :] = title_bar
        combined_img[title_height:title_height+orig_h, :] = orig_img
        
        images.append(combined_img)
        
        # 更新总高度
        total_img_height += combined_img.shape[0] + padding
    
    # 减去最后一个多余的间距
    total_img_height -= padding
    
    # ===== 创建图像区域画布 =====
    # 宽度为最大图片宽度
    img_region_width = max_img_width
    img_region = np.full((total_img_height, img_region_width, 3), bgr_bg_color, dtype=np.uint8)
    
    # 放置图片（垂直排列）
    current_y = 0
    for img in images:
        # 水平居中
        start_x = (img_region_width - img.shape[1]) // 2
        img_region[current_y:current_y+img.shape[0], start_x:start_x+img.shape[1]] = img
        current_y += img.shape[0] + padding
    
    # ===== 文本区域处理 =====
    # 渲染CoT文本
    text_area, text_height = render_text_with_pil(
        cot_text, 
        text_region_width, 
        total_img_height - 60, 
        font_size=24,
        bg_color=bg_color, 
        text_color=text_color
    )
    
    # 添加CoT标题
    cot_title_bar = np.full((60, text_region_width, 3), bgr_bg_color, dtype=np.uint8)
    cot_title_bar_pil = Image.fromarray(cv2.cvtColor(cot_title_bar, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(cot_title_bar_pil)
    
    if cot_title:
        font = get_supported_font(28)
        text_width_pil = draw.textlength(cot_title, font=font)
        text_x = (text_region_width - text_width_pil) // 2
        text_y = (60 - font.size) // 2
        draw.text((text_x, text_y), cot_title, font=font, fill=text_color)
    
    cot_title_bar = cv2.cvtColor(np.array(cot_title_bar_pil), cv2.COLOR_RGB2BGR)
    
    # 组合文本区域
    text_region = np.vstack((cot_title_bar, text_area))
    text_region_height = text_region.shape[0]
    
    # ===== 组合最终图像 =====
    # 计算总宽度和总高度
    total_width = img_region_width + text_region_width
    total_height = max(total_img_height, text_region_height)
    
    # 创建主画布
    canvas = np.full((total_height, total_width, 3), bgr_bg_color, dtype=np.uint8)
    
    # 放置图像区域（左侧）
    canvas[:total_img_height, :img_region_width] = img_region[:total_img_height, :]
    
    # 放置文本区域（右侧）
    canvas[:text_region_height, img_region_width:] = text_region
    
    # ===== 添加主标题 =====
    title_bar_height = 80
    title_bar = np.full((title_bar_height, total_width, 3), bgr_bg_color, dtype=np.uint8)
    title_bar_pil = Image.fromarray(cv2.cvtColor(title_bar, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(title_bar_pil)
    
    if main_title:
        font = get_supported_font(40)
        text_width_pil = draw.textlength(main_title, font=font)
        text_x = (total_width - text_width_pil) // 2
        text_y = (title_bar_height - font.size) // 2
        draw.text((text_x, text_y), main_title, font=font, fill=text_color)
    
    title_bar = cv2.cvtColor(np.array(title_bar_pil), cv2.COLOR_RGB2BGR)
    
    # 将标题添加到画布顶部
    final_canvas = np.vstack((title_bar, canvas))
    
    # ===== 保存结果 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config["result"]["cot_visualization"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cot_visualization_{timestamp}_{base_name}.png")
    
    # 保存为无损PNG（最高质量）
    cv2.imwrite(output_path, final_canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    return output_path

def visual_model_response(image_path, answer_dict, output_path, answer_metainfo, target_size):
    scene = answer_dict["scene"]
    lights = answer_dict["traffic_light"]
    aggregated = aggregate_traffic_light_attributes(lights)
    metainfo = {
        "scene": scene,
        "total_num_of_traffic_lights": len(lights),
        "focusedTrafficLightIndex": [x for x in range(1, len(lights)+1)],
        "bboxes": aggregated["bboxes"],
        "type": ["unknown"] * len(lights),
        "pictogram": ["unknown"] * len(lights),
        "direction": ["unknown"] * len(lights),
        "light_state":  aggregated["light_state"],
        "indication": aggregated["indication"],
        "attachment": ["unknown"] * len(lights)
    }
    visualize_annotation(image_path=image_path, 
                         metainfo=metainfo, 
                         answer_metainfo=answer_metainfo,
                         label_mode="default", 
                         show_scene=False,
                         property_tags=True,
                         output_image_path=output_path,
                         number_label=False,
                         target_size=target_size)


def aggregate_traffic_light_attributes(traffic_light_list):
    """
    将交通灯字典列表中的相同属性聚合到一起
    
    参数:
        traffic_light_list (list): 交通灯字典列表，每个字典包含相同的键
        
    返回:
        dict: 聚合后的属性字典，键为属性名，值为该属性在所有交通灯中的值列表
    """
    # 初始化聚合字典
    aggregated = {
        "bboxes": [],
        "light_state": [],
        "indication": []
    }
    
    if not traffic_light_list:
        return aggregated
    

    # 遍历所有交通灯字典
    for light in traffic_light_list:
        # 提取bbox_2d
        aggregated["bboxes"].append(light["bbox_2d"])
        
        # 提取state
        aggregated["light_state"].append(light["state"])
        
        # 提取indications（保持为列表）
        aggregated["indication"].append(light["indications"])
    
    return aggregated

def embed_original_image(orig_img, container_width, title, bg_color, text_color):
    """
    将原始图片无损嵌入到容器中，添加标题
    返回处理后的图像和实际使用的高度
    """
    # 获取原始图片尺寸
    orig_h, orig_w = orig_img.shape[:2]
    
    # 计算缩放比例（如果需要缩小）
    scale_ratio = 1.0
    if orig_w > container_width:
        scale_ratio = container_width / orig_w
    
    # 如果需要缩放，则调整图片尺寸
    if scale_ratio < 1.0:
        new_w = int(orig_w * scale_ratio)
        new_h = int(orig_h * scale_ratio)
        # 使用高质量缩放算法
        resized_img = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_to_use = resized_img
        use_h, use_w = new_h, new_w
    else:
        img_to_use = orig_img
        use_h, use_w = orig_h, orig_w
    
    # 计算水平居中位置
    start_x = (container_width - use_w) // 2
    
    # 创建容器（高度足够容纳图片和标题）
    container_height = use_h + 50  # 图片高度 + 标题高度
    bgr_bg_color = (bg_color[2], bg_color[1], bg_color[0])
    container = np.full((container_height, container_width, 3), bgr_bg_color, dtype=np.uint8)
    
    # 嵌入图片
    container[20:20+use_h, start_x:start_x+use_w] = img_to_use
    
    # 添加标题
    if title:
        # 创建标题区域
        title_bar = np.full((50, container_width, 3), bgr_bg_color, dtype=np.uint8)
        title_bar_pil = Image.fromarray(cv2.cvtColor(title_bar, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(title_bar_pil)
        
        font = get_supported_font(28)
        text_width_pil = draw.textlength(title, font=font)
        text_x = (container_width - text_width_pil) // 2
        text_y = (50 - font.size) // 2
        draw.text((text_x, text_y), title, font=font, fill=text_color)
        
        title_bar = cv2.cvtColor(np.array(title_bar_pil), cv2.COLOR_RGB2BGR)
        
        # 组合图片和标题
        container[:50, :] = title_bar
    
    return container, container_height

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

def visual_model_result(img1_path, img2_path, img3_path, text_content):

    font_path = '/root/autodl-tmp/data/fonts/Arial Unicode MS.ttf'

    # 检查字体文件是否存在
    if os.path.exists(font_path):
        # 添加字体到字体管理器
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        
        # 设置全局字体
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print(f"已成功加载字体: {prop.get_name()}")
    else:
        print(f"警告: 字体文件未找到: {font_path}")
        # 尝试使用备用字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']

    # 创建画布 - 根据图片尺寸动态调整
    fig = plt.figure(figsize=(20, 15), dpi=120, facecolor='#f5f5f5')  # 增大画布尺寸
    fig.suptitle('专业科学数据可视化分析报告', fontsize=28, weight='bold', y=0.97, color='#1a5276')

    # 创建网格布局：1行2列，宽度比例为3:1
    gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig, wspace=0.2)

    # 左侧图片区域（3行1列） - 增加垂直间距
    left_gs = gs[0].subgridspec(3, 1, hspace=0.5)
    axs = [fig.add_subplot(left_gs[i]) for i in range(3)]

    # 右侧文字区域
    text_ax = fig.add_subplot(gs[1])

    # 示例图片路径（实际使用时替换为您的图片路径）
    image_paths = [
        img1_path,
        img2_path,
        img3_path
    ]

    # 显示三张图片并保持原始宽高比
    for i, ax in enumerate(axs):
        try:
            # 实际使用时取消下面两行注释
            img = mpimg.imread(image_paths[i])
            ax.imshow(img)
            
            img_data = np.random.rand(1080, 1920, 3)  # 1920x1080 RGB图像
            for channel in range(3):
                img_data[:, :, channel] = np.sin(np.linspace(0, 10 * (i+1), 1920)) * np.cos(np.linspace(0, 8 * (i+1), 1080)).reshape(-1, 1)
            
            # 显示图片并保持原始宽高比
            ax.imshow(img_data, aspect='auto')
            
            # 设置图片标题
            ax.set_title(f'高分辨率图像 {i+1} (1920×1080)', fontsize=16, pad=15)
            
            # 美化边框
            for spine in ax.spines.values():
                spine.set_color('#3498db')
                spine.set_linewidth(2.5)
                
            # 添加分辨率标签
            ax.text(0.98, 0.02, '1920×1080', 
                    transform=ax.transAxes, 
                    fontsize=12, 
                    color='white',
                    ha='right', 
                    va='bottom',
                    bbox=dict(facecolor='black', alpha=0.5, pad=4))
                
        except Exception as e:
            print(f"加载图像 {i+1} 时出错: {str(e)}")
            # 显示错误占位符
            ax.text(0.5, 0.5, f"图像 {i+1} 加载失败", 
                    ha='center', va='center', fontsize=18, color='red')
            ax.set_facecolor('#ffecec')


    # 设置文字区域 - 专业布局
    text_ax.set_facecolor('#ffffff')
    text_ax.set_title('分析摘要', fontsize=20, pad=25, weight='bold', color='#2c3e50')

    # 添加半透明背景框
    text_box = plt.Rectangle((0.05, 0.03), 0.9, 0.94, transform=text_ax.transAxes,
                            facecolor='#f8f9fa', edgecolor='#e0e0e0', alpha=0.8, 
                            linewidth=2, zorder=-1, boxstyle='round,pad=0.5')
    text_ax.add_patch(text_box)

    # 添加文字 - 右对齐优化
    text_ax.text(0.5, 0.90, text_content, 
                ha='center',  # 居中显示更美观
                va='top', 
                fontsize=16, 
                color='#2d3436',
                transform=text_ax.transAxes,
                linespacing=1.5)  # 增加行间距

    # 添加水印
    text_ax.text(0.5, 0.5, '内部文件', 
                fontsize=80, 
                color='gray', 
                alpha=0.1,
                ha='center', 
                va='center', 
                rotation=30, 
                weight='bold')

    # 隐藏文字区域的坐标轴
    text_ax.set_axis_off()

    # 调整布局 - 优化间距
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为顶部标题留出空间

    # 保存高分辨率图像（可选）
    plt.savefig('professional_report.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())