"""
生成单条训练数据, 数据格式参考swift文档。
"""
def round_bboxes(bboxes):
    rounded_bboxes = []
    for bbox in bboxes:
        # 如果 bbox 非空，处理每个坐标
        if bbox:  # 确保 bbox 有元素
            x1, y1, x2, y2 = bbox
            # 对每个坐标四舍五入为整数
            x1_int = int(x1 + 0.5)
            y1_int = int(y1 + 0.5)
            x2_int = int(x2 + 0.5)
            y2_int = int(y2 + 0.5)
            rounded_bboxes.append([x1_int, y1_int, x2_int, y2_int])
        else:
            # 如果 bbox 是空列表（例如 []），则跳过或保持空（但根据结构，通常每个元素是4个坐标，这里不会触发）
            pass  # 实际中，bboxes 的元素一般是4个坐标，此分支仅作保护
    return rounded_bboxes

def boxes2string(round_bboxes):
    """
    将bbox坐标转换为用于训练的字符串。
    占位符参考模型文件中的tokenizer_config.json。
    """
    # 使用生成器表达式高效处理每个bbox
    box_strings = (
        f"<|box_start|>[{','.join(map(str, bbox))}]<|box_end|>"
        for bbox in round_bboxes
    )
    # 用逗号和空格连接所有bbox字符串
    return ", ".join(box_strings)

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
            f"<|object_ref_start|>Traffic Light {i+1}<|object_ref_end|> : "
            f"<|box_start|>{int_bbox}<|box_end|>"
        )
        lines.append(line)
    
    # 添加换行符连接所有行
    return "\n".join(lines)

def build_instruction_data(bbox_hint, assistant_content, image_path):

    system_message = "你是一名交通场景分析助手。用户会提供任务描述、推理步骤及必要信息。" \
            "你必须严格按照用户指定的推理步骤，逐步分析并输出结果。" \
            "推理过程和答案需分别包含在 <think> </think> 和 <answer> </answer> 标签内，示例：<think> 此处为推理过程 </think> <answer> 此处为答案（需严格按照用户给的格式输出，如字典格式，字段为指定英文等） </answer>"

    user_message = "<image>\n# 任务：\n识别机动车驾驶员需关注的交通灯（以下简称「关注灯」），输出其灯光状态并推断指示方向。 " \
            "如果图片中存在带倒计时数字的交通灯，需输出关注灯的剩余倒计时（如果有）。\n" \
            "# 请按以下步骤一步步推理：\n" \
            "## Step 1: 判断场景与路口的关系。 \n" \
            "### 提示1: 场景分为四类，分别为：in front of a one-way intersection, in front of a normal/large intersection, on the road, passing the intersection。\n\n" \
            "## Step 2: 识别机动车驾驶员需关注的交通灯。 \n" \
            "### 提示1：逐一识别图片中每个交通灯的属性，如类型、朝向、样式、安装位置等。\n" \
            "### 提示2：根据上述属性分析该交通灯是否需机动车驾驶员关注，用✅表示需关注，❌表示无需关注，带倒计时的灯额外标注⚠️。\n\n" \
            "## Step 3: 输出交通灯状态、指示方向及倒计时数字（如果有倒计时数字）。 \n" \
            "### 提示1：从Step 2的关注灯中识别灯光状态，结合图案样式与状态推断指示方向。\n" \
            "### 提示2：若存在倒计时灯，需分析图片中的倒计时数字归属哪个或哪些交通灯。\n\n" \
            "### 如果交通灯为圆形灯组，那么进行相对位置规则的辅助判断，进一步确定灯态。即以下Step3的分支小步骤：\n" \
            "### [相对位置灯态检测] Step 3.1：观察亮起的子灯。 \n" \
            "#### 提示1：重点关注图像中该交通灯边界框区域，若存在亮着的子灯，那么找出亮度最高的子灯并给出中心位置。\n\n" \
            "### [相对位置灯态检测] Step 3.2：分割灯组。 \n" \
            "#### 提示1：根据灯组排列分割为等高或等宽的三个区域。\n" \
            "#### 提示2：根据Step 3.1中识别的子亮灯的中心位置，判断亮起的子灯位于哪个区域。\n" \
            "#### 提示3：根据分割区域对应灯态的规则，判断子亮灯的颜色，也即灯组展示的灯态。\n\n" \
            "### [相对位置灯态检测] Step 3.3：通过区域规则交叉验证。 \n" \
            "#### 提示1：将Step 3中识别到的灯光状态（初步观察）与Step 3.2中规则判断的灯组灯态相比较。\n\n" \
            "# 输出格式：\n" \
            "<think>\n" \
            "此处填写推理过程。\n" \
            "</think>\n" \
            "<answer>\n" \
            "[\n" \
            '  {"bbox_2d": [x1,y1,x2,y2], "indications": ["left-turn"], "state":"green", "countdown_number": None or Integer Number}\n' \
            '  {"bbox_2d": [x1,y1,x2,y2], "indications": ["U-turn"], "state":"red", "countdown_number": None or Integer Number}\n' \
            "]\n" \
            "</answer>\n" \
            "下面是图片中交通灯的边界框，按顺序逐一列出：\n" \
            f"{round_bboxes(bbox_hint)}"

    entry = {
        "messages": [
            {
                "role":"system",
                "content":system_message
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ],
        "images": [image_path]
    }

    return entry

def build_instruction_data_end2end(assistant_content, image_path):

    system_message = "You are a traffic scene analysis assistant. The user will provide the task description, reasoning steps, and necessary information. " \
            "You must strictly follow the reasoning steps specified by the user, analyze step by step, and output the results. " \
            "The reasoning process and the answer are contained in the <think> </think> and <answer> </answer> tags respectively, for example, <think> The reasoning process is here </think> <answer> The answer is here </answer>"

    user_message = "<image>\n# Task:\nIdentify the traffic lights that the driver needs to pay attention to (hereinafter called attention light). Output their light status and infer the direction they indicate. " \
            "If there is a traffic light with a countdown number in the picture, output the remaining time of the attention light (if any).\n" \
            "# You need to follow the solution step by step:\n" \
            "## Step 1: Determine the relationship between the scene and the intersection.\n" \
            "### Tips 1: The scene are divided into four categories: in front of a one-way intersection, in front of a normal/large intersection, on the road, passing the intersection.\n\n" \
            "## Step 1: Identify traffic lights that drivers should pay attention to.\n" \
            "### Tips 1: Identify the attributes of traffic lights in the image one by one, such as type, orientation, pattern, mounting position, etc.\n" \
            "### Tips 2: Analyze whether it needs to be paid attention to by the driver from the above attributes. Give ✅ and ❌ to distinguish the results, and give ⚠️ to those countdown lights.\n\n" \
            "## Step 2: Outputs the traffic light status, the direction it is indicating, and the countdown number (if any). \n" \
            "### Tips 1: From the traffic lights in step 1, identify the light state, and infer the direction indicated by combining the pattern and the light state.\n" \
            "### Tips 2: If there is a countdown light, analyze which traffic light the countdown numbers in the picture belong to.\n\n" \
            "# Output format:\n" \
            "<think>\n" \
            "The reasoning process is here.\n" \
            "</think>\n" \
            "<answer>\n" \
            "[\n" \
            '  {"bbox_2d": [x1,y1,x2,y2], "indications": ["left-turn"], "state":"green", "countdown_number": None or Integer Number}\n' \
            '  {"bbox_2d": [x1,y1,x2,y2], "indications": ["U-turn"], "state":"red", "countdown_number": None or Integer Number}\n' \
            "]\n" \
            "</answer>\n" \

    entry = {
        "messages": [
            {
                "role":"system",
                "content":system_message
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ],
        "images": [image_path]
    }

    return entry


def build_instruction_data_noCoT(bbox_hint, assistant_content, image_path):

    system_message = "You are a traffic scene analysis assistant. The user will provide the task description and necessary information." \

    user_message = "<image>\n# Task:\nIdentify the traffic lights that the driver needs to pay attention to (hereinafter called attention light). Output their light status and infer the direction they indicate. " \
            "If there is a traffic light with a countdown number in the picture, output the remaining time of the attention light (if any).\n" \
            "# Output format:\n" \
            "<answer>\n" \
            "[\n" \
            '  {"bbox_2d": [x1,y1,x2,y2], "indications": ["left-turn"], "state":"green", "countdown_number": None or Integer Number}\n' \
            '  {"bbox_2d": [x1,y1,x2,y2], "indications": ["U-turn"], "state":"red", "countdown_number": None or Integer Number}\n' \
            "]\n" \
            "</answer>\n" \
            "Below are the bounding boxes of the traffic lights in the picture one by one in order of sequence:\n" \
            f"{round_bboxes(bbox_hint)}"

    entry = {
        "messages": [
            {
                "role":"system",
                "content":system_message
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ],
        "images": [image_path]
    }

    return entry

def build_instruction_data_noCoT_end2end(assistant_content, image_path):

    system_message = "You are a traffic scene analysis assistant. The user will provide the task description, reasoning steps, and necessary information." \

    user_message = "<image>\n# Task:\nIdentify the traffic lights that the driver needs to pay attention to (hereinafter called attention light). Output their light status and infer the direction they indicate. " \
            "If there is a traffic light with a countdown number in the picture, output the remaining time of the attention light (if any).\n" \
            "# Output format:\n" \
            "<answer>\n" \
            "[\n" \
            '  {"bbox_2d": [x1,y1,x2,y2], "indications": ["left-turn"], "state":"green", "countdown_number": None or Integer Number}\n' \
            '  {"bbox_2d": [x1,y1,x2,y2], "indications": ["U-turn"], "state":"red", "countdown_number": None or Integer Number}\n' \
            "]\n" \
            "</answer>\n" \

    entry = {
        "messages": [
            {
                "role":"system",
                "content":system_message
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ],
        "images": [image_path]
    }

    return entry
