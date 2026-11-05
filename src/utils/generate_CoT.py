# generate_CoT.py


import math
import re
from utils.annotation_processing import generate_annswer, countdown_binding
from utils.config_loader import load_config
import json
from utils.trace_analyzer import analyze_vehicle_behavior_from_trajectory,analyze_traj_with_light
# 获取CoT的2种方式：基于模板 template 和 基于 AI
# 将以上二者再进行分类： 是否训练端到端模型
# 基于AI则需要对AI生成的数据进行处理，组合成思维链，还需要检验思维链数据是否合理

# 基于模板
def generate_CoT_by_template(metainfo: dict, traceinfo=None) -> str:  # 新增traceinfo参数
    thinking = generate_thinking_by_template(metainfo=metainfo, traceinfo=traceinfo)
    answer = generate_answer_in_CoT(generate_annswer(metainfo=metainfo))
    CoT = thinking + answer
    return CoT

# 基于模板、端到端
def generate_CoT_by_template_end2end(metainfo: dict, traceinfo=None) -> str:  # 新增traceinfo参数
    thinking = generate_thinking_by_template_end2end(metainfo=metainfo, traceinfo=traceinfo)
    answer = generate_answer_in_CoT(generate_annswer(metainfo=metainfo))
    CoT = thinking + answer
    return CoT

# 基于AI
def generate_CoT_by_ai(metainfo: dict, base_name:str, traceinfo=None) -> str:  # 新增traceinfo参数
    thinking = generate_thinking_by_ai(metainfo, base_name, traceinfo=traceinfo)
    answer = generate_answer_in_CoT(generate_annswer(metainfo))
    CoT = thinking + answer
    return CoT

# 基于AI 端到端
def generate_CoT_by_ai_end2end(metainfo: dict, traceinfo=None):  # 新增traceinfo参数
    thinking = generate_thinking_by_ai_end2end(metainfo, traceinfo=traceinfo)
    answer = generate_answer_in_CoT(generate_annswer(metainfo))
    CoT = thinking + answer
    return CoT

## 基于模板
# 以下Step1到Step3的是具体实现， 比较复杂
def get_step1_by_template(metainfo: dict) -> str:
    if metainfo["scene"] == "in front of a one-way intersection":
        step1 = f"当前场景为 \"{metainfo['scene']}\".\n根据图像内容可推断，当前车辆即将进入单向路口，原因是道路前方的交通灯仅使用圆形灯，无机动车转向箭头；此外，未观察到横穿或垂直方向行驶的车辆。"
    elif metainfo["scene"] == "in front of a normal/large intersection":
        step1 = f"当前场景为 \"{metainfo['scene']}\".\n根据图像内容可观察到，车辆即将进入一个路口。道路前方的交通灯组结构复杂，且能清晰看到车道交叉处或交叉方向行驶的车辆，由此可确认该路口交通形态复杂。"
    elif metainfo["scene"] == "on the road":
        step1 = f"当前场景为 \"{metainfo['scene']}\".\n根据图像内容，因未观察到交通灯等路口相关信息，无法判断车辆是否即将进入路口。结合当前车辆与周边车辆的行驶状态，可推断此场景为道路正常行驶场景。"
    elif metainfo["scene"] == "passing the intersection":
        step1 = f"当前场景为 \"{metainfo['scene']}\".\n从图像可见，当前车辆可能已越过路口前的停止线，但尚未进入下一个路口，说明车辆正处于路口内部。道路上方的交通灯清晰可见。"
    else:
        step1 = "无法识别当前场景。"
    return step1

def get_step2_by_template(metainfo: dict) -> list:
    """
    根据交通信号灯的元信息生成描述文本：
    1. 显示每个灯的边界框；
    2. 仅圆形灯判断排列方式（纵向/横向/无法确定），非圆形灯按原逻辑；
    3. 无法确定排列方式时按原逻辑，确定时提示后续关注亮区。
    
    参数:
        metainfo (dict): 包含交通信号灯信息的字典，需包含以下键：
            - 'focusedTrafficLightIndex': 需关注的信号灯索引列表（1-based）
            - 'total_num_of_traffic_lights': 信号灯总数
            - 'type': 信号灯类型的列表
            - 'pictogram': 图案信息的列表（每个元素为字符串列表）
            - 'attachment': 安装方式的列表
            - 'direction': 朝向的列表

    返回:
        list: 每个信号灯的描述文本列表
    """
    
    # 定义映射字典（完全保留原有逻辑）
    TYPE_DESC = {
        "traffic lights for motor vehicles at intersections": "交叉路口处的机动车交通信号灯",
        "Other traffic lights at intersections": "交叉路口处的非机动车或行人交通信号灯",
        "countdown lights": "带倒计时功能的交通信号灯",
        "lane signal lights": "指示车道专用通行指令的交通信号灯"
    }

    ATTACHMENT_DESC = {
        "horizontal bar": "附着在水平横杆上",
        "vertical poles": "安装在垂直立柱上",
        "bridge deck": "设置在桥面路面上"
    }

    DIRECTION_DESC = {
        "front": "朝向驾驶员方向",
        "left": "朝向左侧方向",
        "right": "朝向右侧方向",
        "back": "背离驾驶员方向"
    }

    # 新增：判断圆形灯组的排列方式（纵向/横向/无法确定），宽高比阈值1.2
    def _judge_circular_group_orientation(bbox):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return "无法确定（边界框尺寸无效）"
        # 宽高比阈值1.2
        if height > width * 1.2:
            return "纵向排列（高度 > 宽度 × 1.2，含子灯的完整灯组）"
        elif width > height * 1.2:
            return "横向排列（宽度 > 高度 × 1.2，含子灯的完整灯组）"
        else:
            return "无法确定（宽高比 < 1.2，大概率为单个圆形灯）"

    # 原有子函数（完全保留）

    def _get_type_description(type_val):
        return TYPE_DESC.get(type_val, "未知类型")

    def _get_attachment_description(attachment_val):
        return ATTACHMENT_DESC.get(attachment_val, "安装方式未知")

    def _get_direction_description(direction_val):
        return DIRECTION_DESC.get(direction_val, "朝向不确定")

    def _get_pictogram_description(pictogram_vals):
        if not pictogram_vals or "unknown" in pictogram_vals:
            return "显示无法识别的标识"
        
        arrows = [p.replace("-arrow", "").title() for p in pictogram_vals if "arrow" in p]
        if len(arrows) > 1:
            return f"显示多个箭头（如： {', '.join(arrows)}）"
        elif arrows:
            return f"显示 {arrows[0]} 向箭头"
        
        if "circle" in pictogram_vals:
            return "显示圆形灯（可能是包含红 / 黄 / 绿子灯的灯组）"
        if "pedestrian" in pictogram_vals:
            return "显示行人标识"
        if "non-motor-vehicle" in pictogram_vals:
            return "显示非机动车标识"
        if "countdown" in pictogram_vals:
            return "显示倒计时标识"
        
        return "显示无法识别的标识"

    def _get_relevance_description(idx, type_val):
        pictogram = metainfo["pictogram"][idx]
        if (idx + 1) in focusedTrafficLightIndex:
            if ("countdown" in pictogram) & (type_val!="Other traffic lights at intersections"):
                countdown_number = next(int(x) for x in pictogram if x.isdigit()) if any(x.isdigit() for x in pictogram) else None
                if countdown_number:
                    return f"这是驾驶员需要关注的信号灯。注意它显示的倒计时数字为 {countdown_number}。 ✅⚠️"
                else:
                    return "这是驾驶员需要关注的信号灯。但该信号灯上的倒计时数字无法识别。 ✅⚠️"
            else:
                return "这是驾驶员需要关注的信号灯。 ✅"
        else:
            if ("countdown" in pictogram) & (type_val!="Other traffic lights at intersections"):
                countdown_number = next(int(x) for x in pictogram if x.isdigit()) if any(x.isdigit() for x in pictogram) else None
                if countdown_number:
                    return f"这不是驾驶员需要关注的信号灯。但该信号灯有倒计时数字 {countdown_number}，我需要在步骤3中根据它推断其他信号灯的剩余时间。 ❌⚠️"
                else:
                    return "这不是驾驶员需要关注的信号灯。且该信号灯上的倒计时数字无法识别。 ❌⚠️"
            else:
                return "这不是驾驶员需要关注的信号灯。 ❌"

    try:
        descriptions = []
        focusedTrafficLightIndex = metainfo["focusedTrafficLightIndex"]
        total_num = metainfo["total_num_of_traffic_lights"]
        # 新增：获取边界框列表，校验长度一致性
        bboxes = metainfo.get("bboxes", [])
        if len(bboxes) != total_num:
            raise ValueError(f"'bboxes'的长度 ({len(bboxes)}) 与 'total_num_of_traffic_lights' ({total_num}) 不一致")

        for idx in range(total_num):
            idx_plus_1 = idx + 1
            type_val = metainfo["type"][idx]
            pictogram_vals = metainfo["pictogram"][idx]
            attachment_desc = _get_attachment_description(metainfo["attachment"][idx])
            direction_desc = _get_direction_description(metainfo["direction"][idx])
            pictogram_desc = _get_pictogram_description(pictogram_vals)
            relevance_desc = _get_relevance_description(idx, type_val)
            # 新增：获取当前灯的边界框，处理坐标格式
            bbox = bboxes[idx]
            int_bbox = [int(round(coord)) for coord in bbox]
            bbox_base_desc = f"其边界框为 {int_bbox}。"

            # 核心逻辑：判断是否是圆形灯，再处理排列方式
            is_circular = "circle" in pictogram_vals
            if not is_circular:
                # 非圆形灯：按原逻辑，仅补充边界框描述
                description = (
                    f"Traffic Light {idx_plus_1}: 这是一个 {_get_type_description(type_val)}， {attachment_desc}。"
                    f"{bbox_base_desc}"  # 仅新增边界框
                    f"它 {direction_desc} 而且 {pictogram_desc}。 {relevance_desc}"
                    )
            else:
                # 圆形灯：判断排列方式
                orientation = _judge_circular_group_orientation(bbox)
                if "undetermined" in orientation:
                    # 无法确定排列方式：按原逻辑，补充边界框和排列方式说明
                    description = (
                        f"Traffic Light {idx_plus_1}: 这是一个 {_get_type_description(type_val)}的交通灯，{attachment_desc}。"
                        f"{bbox_base_desc}"
                        f"根据边界框比例（阈值1.2），该灯的排列方式为 {orientation} （大概率为单个圆形灯）。"
                        f"它 {direction_desc} 而且 {pictogram_desc}。 {relevance_desc}"
                    )
                else:
                    # 确定纵向/横向：说明是整个灯组，提示后续关注亮区
                    description = (
                        f"Traffic Light {idx_plus_1}: 这是一个 {_get_type_description(type_val)}的交通灯，{attachment_desc}。"
                        f"{bbox_base_desc}"
                        f"根据边界框比例（阈值1.2），该灯的排列方式为 {orientation}。"
                        f"注意：这是整个灯组（包含红灯/黄灯/绿灯子灯），而不是单个子灯；"
                        f"我将在Step 3.1和Step 3.2中关注该区域，以识别亮着的子灯。"
                        f"它 {direction_desc} 而且 {pictogram_desc}。 {relevance_desc}"
                    )
            descriptions.append(description)

        return descriptions

    except KeyError as e:
        raise ValueError(f"元信息中缺少必要的键：{e}") from e
    except IndexError as e:
        raise ValueError("元信息列表的索引超出范围") from e


def get_step3_by_template(metainfo: dict) -> list:
    """
    基于模板生成Step 3的Chain-of-Thought推理过程：
    1. 所有场景均保留“初步灯态结果 + 倒计时 + 方向指示”的原有完整逻辑；
    2. 仅圆形灯组（纵向/横向）新增亮区推理文本，用于优化灯态判断（不替代原有逻辑）。
    """
    # 原有子函数（完全保留，确保基础逻辑不变）
    def _get_countdown_info(light_idx, countdown_dict):
        """生成倒计时关联信息（原有逻辑）"""
        for main_id, linked_ids in countdown_dict.items():
            try:
                if int(main_id) == linked_ids[0]:
                    pictogram = metainfo["pictogram"][int(main_id) - 1]
                    countdown_number = next(int(x) for x in pictogram if x.isdigit()) if any(x.isdigit() for x in pictogram) else None
                    return f" 该交通信号灯自身可显示倒计时，当前显示的倒计时数字为 {countdown_number}。" if countdown_number else ""
                else:
                    if int(light_idx + 1) in linked_ids:
                        try:
                            pictogram = metainfo["pictogram"][int(main_id) - 1]
                            countdown_number = next(int(x) for x in pictogram if x.isdigit()) if any(x.isdigit() for x in pictogram) else None
                            return (
                                f"Traffic Light {main_id} 是与 Traffic Light {light_idx + 1}集成的倒计时装置，两者保持同步状态。 "
                                f"由于系统显示其倒计时数字为 {countdown_number} ，因此 Traffic Light {light_idx + 1}的倒计时也为{countdown_number}秒。"
                            ) if countdown_number else ""
                        except (IndexError, StopIteration, ValueError):
                            continue
            except:
                continue
        return ""

    def _get_pictogram_description(pictogram_vals, light_state):
        """生成交通信号灯图案的详细描述（原有逻辑，支撑灯态和图案属性） 
        参数:
            pictogram_vals: 信号灯图案列表，可能包含以下值：
                "left-arrow", "right-arrow", "U-turn-arrow", "straight-arrow"
                "circle", "countdown", "non-motor-vehicle", "pedestrian", "unknown"
                数字字符串 (如 "5") - 通常与"countdown"同时出现
            light_state: 灯光状态 (如 'red', 'flashing green', 'off')
        
        返回:
            自然语言描述的字符串
        """
        if "off" in light_state:
            if pictogram_vals:
                pattern_desc = _describe_patterns(pictogram_vals)
                return f"该交通信号灯当前处于关闭状态，但可见图案为 {pattern_desc}"
            return "该交通信号灯当前处于关闭状态"
        
        if pictogram_vals:
            pattern_desc = _describe_patterns(pictogram_vals)
            if "flashing" in light_state:
                color = light_state.replace("flashing ", "")
                return f"该交通信号灯正在闪烁 {color} 灯光，带有 {pattern_desc}"
            return f"该交通信号灯显示 {light_state} 灯光，带有 {pattern_desc}"
        
        if "flashing" in light_state:
            return f"一盏纯色 {color} 灯正在闪烁"
        
        return f"显示一盏纯色 {light_state} 灯"

    def _describe_patterns(patterns):
        """描述一个或多个信号灯图案（原有逻辑，支撑图案属性）特别处理数字倒计时
        
        参数:
            patterns: 图案字符串列表
            
        返回:
            自然语言描述的图案组合
        """
        # 1. 分类处理不同类型的图案
        arrows = [p for p in patterns if "arrow" in p]
        circles = [p for p in patterns if p == "circle"]
        countdowns = [p for p in patterns if p == "countdown"]
        pedestrians = [p for p in patterns if p == "pedestrian"]
        non_motor = [p for p in patterns if p == "non-motor-vehicle"]
        unknowns = [p for p in patterns if p == "unknown"]
        
        # 2. 特别处理数字倒计时
        # 寻找数字字符串（如"5"）
        number_strs = [p for p in patterns if p.isdigit()]
        
        # 组合倒计时描述
        countdown_desc = ""
        if countdowns:
            if number_strs:
                # 使用第一个找到的数字（通常只有一个）
                countdown_desc = f"显示 {number_strs[0]} 秒的倒计时器"
            else:
                countdown_desc = "倒计时器（时间不明确）"
                
        # 3. 收集所有图案描述
        all_descriptions = []
        
        # 处理箭头
        if arrows:
            arrow_types = [arrow.replace("-arrow", "") for arrow in arrows]
            if len(arrow_types) == 1:
                all_descriptions.append(f" {arrow_types[0]} 向箭头")
            else:
                arrow_desc = " 和 ".join(f"{arrow}" for arrow in arrow_types)
                all_descriptions.append(f"分别指向 {arrow_desc} 的箭头")
        
        # 添加其他图案描述
        if circles:
            all_descriptions.append("圆形标识")
        if pedestrians:
            all_descriptions.append("行人标识")
        if non_motor:
            all_descriptions.append("非机动车标识")
        if countdown_desc:  # 使用组合后的倒计时描述
            all_descriptions.append(countdown_desc)
        if unknowns:
            all_descriptions.append("无法识别的图案")
        
        # 4. 处理未分类的数字（没有countdown的数字）
        unclassified_numbers = [num for num in number_strs if num not in patterns or "countdown" not in patterns]
        for num in unclassified_numbers:
            all_descriptions.append(f"数字 '{num}'")
        
        # 5. 根据图案数量生成描述
        if not all_descriptions:
            return "无法识别的图案"
        
        if len(all_descriptions) == 1:
            return all_descriptions[0]
        
        if len(all_descriptions) == 2:
            return f"{all_descriptions[0]} 和 {all_descriptions[1]}"
        
        # 多个图案组合描述
        return f"{'， '.join(all_descriptions[:-1])}， 和 {all_descriptions[-1]}"

    # 新增：判断当前圆形灯是否为纵向/横向灯组（复用Step2逻辑，仅用于触发推理）
    def  _is_valid_circular_group(bbox, pictogram_vals):
        """返回 (是否是有效灯组, 排列方式)，仅圆形灯组触发推理"""
        if not any(p in pictogram_vals for p in ["circle", "left-arrow", "right-arrow", "U-turn-arrow", "straight-arrow"]):
        # 当pictogram_vals中没有上述任何一个值时，执行说明
            flag1 = "子灯图案不是圆形，也不是箭头，跳过辅助规则"
            #print("图案列表:",pictogram_vals)
            #print(flag1)
            return (False, None, flag1)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            flag2 = "宽度或高度小于等于0，跳过辅助规则"
            return (False, None, flag2)
        # 宽高比阈值1.2，与Step2一致
        if height > width * 1.2:
            return (True, "vertical", None)
        elif width > height * 1.2:
            return (True, "horizontal", None)
        else:
            flag3 = "宽高比低于阈值，跳过辅助规则"
            return (False, None, flag3)

    # 生成圆形灯组的亮区推理文本（仅优化灯态判断，不替代原有逻辑）
    def _generate_circular_group_reasoning(bbox, orientation, initial_light_state):
        """生成亮区推理文本，用于交叉验证灯态，保留原有灯态描述"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        

        # 有亮灯
        if initial_light_state != 'off':
            # 确定区域和中心点坐标的映射关系
            if orientation == "vertical":
                third_h = height / 3
                # 纵向灯组：红在上，黄在中，绿在下
                region_mapping = {
                    'red': ('上部', y1 + third_h / 2),
                    'yellow': ('中部', y1 + third_h + third_h / 2),
                    'green': ('下部', y1 + 2 * third_h + third_h / 2)
                }
                # 获取当前灯态对应的区域和y坐标
                region, center_y = region_mapping.get(initial_light_state, ('未知', (y1 + y2) / 2))
                # 计算x坐标（灯组水平中心）
                center_x = (x1 + x2) / 2
            else:  # horizontal
                third_w = width / 3
                # 横向灯组：红在左，黄在中，绿在右
                region_mapping = {
                    'red': ('左部', x1 + third_w / 2),
                    'yellow': ('中部', x1 + third_w + third_w / 2),
                    'green': ('右部', x1 + 2 * third_w + third_w / 2)
                }
                # 获取当前灯态对应的区域和x坐标
                region, center_x = region_mapping.get(initial_light_state, ('未知', (x1 + x2) / 2))
                # 计算y坐标（灯组垂直中心）
                center_y = (y1 + y2) / 2
            
            # 格式化中心点坐标（四舍五入为整数）
            sub_coordinate = f"（{int(round(center_x))}, {int(round(center_y))}）"
            
            # Step2.1: 观察亮区（明确关联灯组边界框）
            step1 = (
                "### [相对位置灯态检测] Step 3.1：观察亮起的子灯\n"
                f"该灯组（圆形或箭头图案）区域位于（边界框 {[int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]}）。"
                f"存在子亮灯，中心位置位于{sub_coordinate}。\n"
            )
            # Step2.2: 分割灯组（纵向y轴/横向x轴，与Step2排列方式匹配）
            if orientation == "vertical":
                step2 = (
                    f"### [相对位置灯态检测] Step 3.2：分割纵向灯组\n"
                    f"该灯组（高度：{round(height)} 像素）被分割为 3 个相等的纵向区域：\n"
                    f"- 上部区域：y ∈ [{round(y1)} ~ {round(y1 + third_h)}]（对应红灯）\n"
                    f"- 中部区域：y ∈ [{round(y1 + third_h)} ~ {round(y1 + 2*third_h)}]（对应黄灯）\n"
                    f"- 下部区域：y ∈ [{round(y1 + 2*third_h)} ~ {round(y2)}]（对应绿灯）\n"
                    f"最亮的子灯位于{region}区域，根据规则，该子亮灯为{initial_light_state}色，该灯组展示为{initial_light_state}灯。\n"
                )
            else:  # horizontal
                step2 = (
                    f"### [相对位置灯态检测] Step 3.2：分割横向灯组\n"
                    f"该灯组（宽度：{round(width)} 像素）被分割为 3 个相等的横向区域：\n"
                    f"- 左部区域：x ∈ [{round(x1)} ~ {round(x1 + third_w)}]（对应红灯）\n"
                    f"- 中部区域：x ∈ [{round(x1 + third_w)} ~ {round(x1 + 2*third_w)}]（对应黄灯）\n"
                    f"- 右部区域：x ∈ [{round(x1 + 2*third_w)} ~ {round(x2)}]（对应绿灯）\n"
                    f"最亮的子灯位于{region}区域，根据规则，该子亮灯为{initial_light_state}色，该灯组展示为{initial_light_state}灯。\n"
                )
            
            # Step2.3: 应用规则交叉验证（明确关联初步灯态）
            step3 = (
                "### [相对位置灯态检测] Step 3.3：通过区域规则交叉验证\n"
                f"初步观察到的灯态为{initial_light_state}——规则判断的灯态为{initial_light_state}。\n"
            )
        # 没有亮灯，即标注里是off
        else:
            # Step2.1: 观察亮区（明确关联灯组边界框）
            step1 = (
                "### [相对位置灯态检测] Step 3.1：观察亮起的子灯\n"
                f"该圆形灯组区域位于（边界框 {[int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]}）。"
                "经过观察，没有发现有亮着的子灯。\n"
            )
            # Step2.2: 分割灯组（纵向y轴/横向x轴，与Step2排列方式匹配）
            if orientation == "vertical":
                step2 = (
                    f"### [相对位置灯态检测] Step 3.2：分割纵向灯组\n"
                    f"该灯组（高度：{round(height)} 像素）为纵向排列。由于没有亮灯，不进行分割。\n"
                    f"故该灯组展示为{initial_light_state}灯。\n"
                )
            else:  # horizontal
                step2 = (
                    f"### [相对位置灯态检测] Step 3.2：分割横向灯组\n"
                    f"该灯组（高度：{round(height)} 像素）为横向排列。由于没有亮灯，不进行分割。\n"
                    f"故该灯组展示为{initial_light_state}灯。\n"
                )
            
            # Step2.3: 应用规则交叉验证（明确关联初步灯态）
            step3 = (
                "### [相对位置灯态检测] Step 3.3：通过区域规则交叉验证\n"
                f"初步观察到的灯态为{initial_light_state}——规则判断的灯态为{initial_light_state}。\n"
            )
        
        return f"\n{step1}\n{step2}\n{step3}"

    try:
        descriptions = []
        answer_indices = [int(i) for i in metainfo["focusedTrafficLightIndex"]]
        countdown_dict = countdown_binding(metainfo)
        # 预定义方向映射（原有逻辑，支撑指示方向属性）
        DIRECTION_MAP = {
            "直行": "straight",
            "左转弯": "left turn",
            "右转弯": "right turn",
            "掉头": "U-turn"
        }
        bboxes = metainfo.get("bboxes", [])
        if len(bboxes) != metainfo["total_num_of_traffic_lights"]:
            raise ValueError(f"'bboxes'的长度 ({len(bboxes)}) 与 'total_num_of_traffic_lights' ({len(metainfo['total_num_of_traffic_lights'])}) 不一致")

        for idx in range(metainfo["total_num_of_traffic_lights"]):
            # 仅分析需关注的灯（原有逻辑）
            if (idx + 1) not in answer_indices:
                continue
            try:
                # 提取基础属性（支撑4个核心属性的源头数据）
                pictogram_vals = metainfo["pictogram"][idx]
                initial_light_state = metainfo["light_state"][idx]  # 初步灯态（后续会优化）
                indication_vals = metainfo["indication"][idx]       # 指示方向
                bbox = bboxes[idx]                                 # 边界框
                # 判断是否为圆形灯组（仅此类场景新增推理）
                is_valid_group, orientation, flag_text = _is_valid_circular_group(bbox, pictogram_vals)

                # 1. 生成原有完整逻辑的描述（支撑4个属性，完全保留）
                # 1.1 图案+初步灯态描述（支撑灯态、图案属性）
                pictogram_desc = _get_pictogram_description(pictogram_vals, initial_light_state)
                # 1.2 倒计时描述（支撑倒计时属性）
                countdown_info = _get_countdown_info(idx, countdown_dict)
                # 1.3 方向指示描述（支撑指示方向属性）
                if "unknown" in indication_vals:
                    indication_desc = "无法推断该交通信号灯指示的方向"
                else:
                    directions = [DIRECTION_MAP.get(ind, ind) for ind in indication_vals]
                    indication_desc = f"根据交通信号灯显示的图案，可以推断其指示的方向为 {'， '.join(directions)}"

                # 2. 保留原有基础部分，按情况添加优化推理
                # 生成基础描述
                base_description = (
                    f"Traffic Light {idx + 1}: 从图中可以看到 {pictogram_desc}。因此，灯光状态为 {initial_light_state}。"
                    f"{countdown_info} "  # 保留倒计时
                    f"{indication_desc}。"  # 保留方向指示（原有逻辑完整闭环）
                )

                if is_valid_group and orientation:
                    # 圆形灯组：基础描述 + 新增亮区推理（优化灯态）
                    reasoning_text = _generate_circular_group_reasoning(bbox, orientation, initial_light_state)
                    description = f"{base_description}{reasoning_text}"  # 添加推理文本
                else:
                    # 非圆形灯/无法确定：使用基础描述，添加推理不启用说明文本
                    description = f"{base_description}{flag_text}"

                descriptions.append(description)

            except IndexError as e:
                raise ValueError(f"Invalid index in metainfo lists: {e}") from e

        return descriptions

    except KeyError as e:
        raise ValueError(f"metainfo中缺少必要的键： {e}") from e

def get_step4_by_trace_template(metainfo: dict, traceinfo=None) -> str:
    """
    轨迹分析推理步骤，返回自然语言描述（整合自CoT2.py）
    """
    if traceinfo is None:
        return ""
    behavior_reasoning = ""
    for idx_1based in metainfo["focusedTrafficLightIndex"]:
        idx = idx_1based - 1
        # 构建单灯信息字典，适配analyze_traj_with_light函数入参
        light_info = {
            "light_state": metainfo["light_state"][idx],
            "pictogram": metainfo["pictogram"][idx],
            "indication": metainfo["indication"][idx],
            "scene": metainfo["scene"],
            "is_attention": True
        }
        # 特殊场景处理：车辆正在通过路口时，跳过单灯分析
        if light_info["scene"] == "passing the intersection":
            behavior_reasoning += "车辆正在通过路口，因此不适用单个交通信号灯的轨迹分析。\n"
            break
        # 调用轨迹分析工具函数，生成单灯推理文本
        single_reasoning = analyze_traj_with_light(traceinfo, light_info)
        behavior_reasoning += f"\n[ Traffic Light {idx_1based} 的分析]\n{single_reasoning}\n"
    # 对推理文本进行Token化替换
    return behavior_reasoning

# 合并 step1到step4， 得到完整CoT
def merge_analysis_steps(step1, step2, step3, step4=None):
    merged = "<think>\n"
    merged += "好的，我需要识别出机动车驾驶员应关注的交通信号灯，输出其状态并推断指示方向。\n如果图片中有带倒计时数字的信号灯，需输出关注信号灯的剩余时间（如有）。 \n"
    # 新增轨迹分析描述（仅当step4非空时显示）
    if step4 is not None and step4 != "":
        merged += "除此之外，若存在车辆轨迹，我需要分析交通信号灯及其与车辆轨迹的关系。\n"
    merged += "为完成此任务，我将其分解为以下步骤：\n"

    # Step 1
    merged += "# Step 1: 确定场景与交叉路口的关系。\n"
    merged += step1 + "\n\n"

    # Step 2
    merged += "# Step 2: 识别机动车驾驶员应关注的交通信号灯。\n"
    merged += "图像中共有 {} 个交通信号灯。\n我将逐一检查这些信号灯，对驾驶员应关注的信号灯标记✅，无需关注的标记❌，带倒计时的标记⚠️。\n".format(len(step2))
    for item in step2:
        merged += "- " + item + "\n"
    merged += "\n"

    # Step 3
    merged += "# Step 3:输出交通信号灯的状态、指示方向及倒计时数字（如有）。\n"
    if len(step3)>0:
        merged += "在Step 2中，我发现有 {} 个驾驶员需要关注的交通信号灯。接下来，我将逐一分析这 {} 个信号灯。\n".format(len(step3), len(step3))
    else:
        merged += "在Step 2中，我发现有 {} 个驾驶员需要关注的交通信号灯。\n".format(len(step3))
    for item in step3:
        merged += "- " + item + "\n"

    # Step 4
    if step4 is not None and step4 != "":
        merged += "\n# Step 4: 分析车辆轨迹及其与交通信号灯的关系。\n"
        merged += step4 + "\n\n"

    merged += "</think>\n"
    return merged

# metainfo -> CoT
def generate_thinking_by_template(metainfo: dict, traceinfo=None) -> str:
    step1 = get_step1_by_template(metainfo)
    step2 = get_step2_by_template(metainfo)
    step3 = get_step3_by_template(metainfo)
    step4 = get_step4_by_trace_template(metainfo, traceinfo)
    thinking = merge_analysis_steps(step1, step2, step3, step4)
    return thinking

# 生成答案，这个代码任何模式都可以用。无论是否端到端，是否有CoT
def generate_answer_in_CoT(answer: dict) -> str:
    processed_lights = []
    for light in answer['traffic_lights']:
        # 四舍五入处理bbox_2d坐标
        rounded_bbox = [int(round(x)) for x in light['bbox_2d']]
        
        # 构建交通灯对象字典
        light_dict = {
            'bbox_2d': rounded_bbox,
            'indications': light['indications'],
            'state': light['state'],
            'countdown_number': light['countdown_number']
        }
        
        # 将字典转换为一行JSON字符串
        json_str = json.dumps(light_dict)
        processed_lights.append(f"\t{json_str}")
    
    # 构建最终输出 - 修复逗号问题
    output_lines = [
        "<answer>",
        "["
    ]
    
    # 添加逗号分隔的数组元素
    for i, light in enumerate(processed_lights):
        # 最后一个元素不加逗号
        if i < len(processed_lights) - 1:
            output_lines.append(f"{light},")
        else:
            output_lines.append(light)
    
    output_lines.append("]")
    output_lines.append("</answer>")
    
    return "\n".join(output_lines)

# 合并step，端到端的情况，区别就是step2需要增加识别到的bbox框，step4加入轨迹，然后开始分析
def merge_analysis_steps_end2end(metainfo, step1, step2, step3, step4=None):
    
    merged = "<think>\n"
    merged += "OK, I need to identify the traffic light that the driver should focus on. "
    merged += "Output the state of the light and infer the direction it indicates. If there is a traffic light with a countdown number in the picture, output the remaining time of the attention light (if any).\n"
    # 新增轨迹分析描述（仅当step4非空时显示）
    if step4 is not None and step4 != "":
        merged += "Overall, if vehicle trajectory exists, I need to analyze the traffic lights and their relationship with the vehicle trajectory.\n"
    merged += "To accomplish this task, I need to break it down into following steps:\n"

    # Step 2
    merged += "# Step 1: Identify traffic lights that drivers should pay attention to. \n"
    merged += "There are {} traffic lights in the image. I will check the traffic lights one by one. And give ✅ to those traffic lights that drivers should pay attention to, ❌ to those traffic lights that drivers don't need to pay attention to, and ⚠️ to those with countdown.\n".format(len(step2))
    for item in step2:
        merged += "- " + item + "\n"
    merged += "\n"

    # Step 3
    merged += "# Step 2: Outputs the traffic light status, the direction it is indicating, and the countdown number (if any).\n"
    if len(step3)>0:
        merged += "In the second step, I found {} traffic lights that drivers need to pay attention to. Next, I will analyze these {} traffic lights one by one.\n".format(len(step3), len(step3))
    else:
        merged += "In the second step, I found {} traffic lights that drivers need to pay attention to.\n".format(len(step3))
    for item in step3:
        merged += "- " + item + "\n"

    # Step 4
    if step4 is not None and step4 != "":
        merged += "\n# Step 3: Analyze the vehicle trajectory and its relationship with the traffic lights.\n"
        merged += step4 + "\n\n"

    merged += "</think>\n"
    return replace_traffic_light_refs(merged)

# 基于ai的生成
def get_step1_by_ai(base_name:str) -> str:
    config = load_config()
    json_path = config["description"]["scene"]
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    scene_dict = {item["base_name"]: item["description"] for item in data}
    return scene_dict[base_name]

def get_step2_by_ai(metainfo: dict, base_name:str) -> list:
    num = metainfo['total_num_of_traffic_lights']
    config = load_config()
    json_path = config["description"]["traffic_light"]
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    light_dict = {item["base_name"]: item["description"] for item in data}
    des = extract_sorted_traffic_lights(light_dict[base_name])
    for index in range(num):
        if  int(index+1) in metainfo["focusedTrafficLightIndex"]:
            des[index] += "✅"
        else:
            des[index] += "❌"

        if metainfo["type"][index] == "countdown lights" or ("countdown" in metainfo["pictogram"][index]):
            des[index] += "⚠️"
    return des
                            
# 修改：新增traceinfo参数，调用Step4
def generate_thinking_by_ai(metainfo: dict, base_name:str, traceinfo=None) -> str:
    step1 = get_step1_by_ai(base_name)
    step2 = get_step2_by_ai(metainfo, base_name)
    step3 = get_step3_by_template(metainfo)
    step4 = get_step4_by_trace_template(metainfo, traceinfo)  # 新增Step4调用
    thinking = merge_analysis_steps(step1, step2, step3, step4)
    return thinking

def extract_sorted_traffic_lights(text):
    # 常见总结词列表（可扩展）
    summary_keywords = [
        "in summary", "to summarize", "overall", "in conclusion",
        "finally", "to sum up", "in short", "in brief", "ultimately",
        "summary", "conclusion", "key points"
    ]
    
    # 构建正则表达式模式 - 更健壮的边界检测
    summary_pattern = "|".join(re.escape(kw) for kw in summary_keywords)
    pattern = rf"Traffic Light (\d+):\s*(.*?)(?=(Traffic Light \d+:|[\n\s]*###|{summary_pattern}|$))"
    
    # 查找所有匹配项
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    # 提取并存储交通灯信息
    light_dict = {}
    for match in matches:
        light_num = int(match[0])
        content = match[1].strip()
        
        # 清理内容
        cleaned = re.sub(r"\*\*|#", "", content)  # 移除加粗和#号
        cleaned = re.sub(r"\s*\n\s*", " ", cleaned)  # 将换行符替换为空格
        cleaned = re.sub(r"\s+", " ", cleaned).strip()  # 合并多余空格
        cleaned = re.sub(r"^\s*:\s*", "", cleaned)  # 移除开头的冒号
        
        light_dict[light_num] = f"Traffic Light {light_num}: {cleaned}"
    
    # 按编号排序
    sorted_lights = [light_dict[num] for num in sorted(light_dict)]
    return sorted_lights

# 修改：新增traceinfo参数，调用端到端的合并函数（含Step4）
def generate_thinking_by_template_end2end(metainfo: dict, traceinfo=None) -> str:
    #step1 = get_step1_by_template(metainfo)
    step2 = get_step2_by_template(metainfo)
    step3 = get_step3_by_template(metainfo)
    step4 = get_step4_by_trace_template(metainfo, traceinfo)  # 新增Step4调用
    thinking = merge_analysis_steps_end2end(metainfo, step2, step3, step4)
    return thinking

def replace_traffic_light_refs(text):
    """
    识别并替换文本中所有 'Traffic Light X' 格式的字符串（X为数字）
    替换格式：<|object_ref_start|>Traffic Light X<|object_ref_end|>
    
    参数:
        text (str): 输入的原始文本
        
    返回:
        str: 替换后的文本
    """
    pattern = r'Traffic Light \d+'
    return re.sub(
        pattern,
        lambda m: f"<|object_ref_start|>{m.group(0)}<|object_ref_end|>",
        text
    )

# 修改：新增traceinfo参数（预留接口，后续可补充AI端到端的轨迹分析逻辑）
def generate_thinking_by_ai_end2end(metainfo, traceinfo=None):
    # 后续可补充：基于AI生成端到端场景的Step4轨迹分析逻辑
    step1 = get_step1_by_ai(base_name=metainfo.get("base_name", ""))  # 假设metainfo含base_name
    step2 = get_step2_by_ai(metainfo, base_name=metainfo.get("base_name", ""))
    step3 = get_step3_by_template(metainfo)
    step4 = get_step4_by_trace_template(metainfo, traceinfo)  # 复用模板的Step4逻辑
    thinking = merge_analysis_steps_end2end(metainfo, step1, step2, step3, step4)
    return thinking
    # pass  # yiyang原有的，不知道什么意思

def generate_traffic_light_references(bbox_list):
    """
    生成交通灯对象引用和边界框的格式化字符串
    
    参数:
        bbox_list (list): 边界框列表，每个元素为 [x1, y1, x2, y2]
        
    返回:
        str: 格式化后的多行字符串
    """
    if not bbox_list:
        return ""
    
    # 生成每行字符串
    lines = []
    for i, bbox in enumerate(bbox_list):
        # 确保bbox是整数列表
        int_bbox = [int(coord) for coord in bbox]
        
        # 格式化字符串
        line = (
            f"Traffic Light {i+1} : "
            f"<|box_start|> {int_bbox} <|box_end|>"
        )
        lines.append(line)
    
    # 添加换行符连接所有行
    return "\n".join(lines)