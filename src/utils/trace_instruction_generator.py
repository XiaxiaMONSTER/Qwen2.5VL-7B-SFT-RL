def round_bboxes(bboxes):
    rounded_bboxes = []
    for bbox in bboxes:
        if bbox:
            x1, y1, x2, y2 = bbox
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
    将边界框列表转为带模型专用Token的逗号分隔字符串（备用）
    参数:
        round_bboxes (list): 经round处理后的整数边界框列表
    返回:
        str: 带<|box_start|>等Token的紧凑字符串
    """
    if not round_bboxes:
        return ""
    box_strings = (
        f"<|box_start|>[{','.join(map(str, bbox))}]<|box_end|>"
        for bbox in round_bboxes
    )
    return ", ".join(box_strings)

def generate_traffic_light_references(bbox_list):
    """
    生成交通灯对象引用和边界框的格式化字符串（带模型专用Token）
    参数:
        bbox_list (list): 经round处理后的整数边界框列表，每个元素为 [x1, y1, x2, y2]
    返回:
        str: 带<|object_ref_start|>等Token的多行字符串
    """
    if not bbox_list:
        return ""
    
    lines = []
    for i, bbox in enumerate(bbox_list):
        # 确保bbox为整数（脚本1已有round_bboxes处理，此处二次校验）
        int_bbox = [int(coord) for coord in bbox]
        line = (
            f"<|object_ref_start|>Traffic Light {i+1}<|object_ref_end|> : "
            f"<|box_start|>{int_bbox}<|box_end|>"
        )
        lines.append(line)
    return "\n".join(lines)

def build_instruction_data(bbox_hint, cot, image_path, summary_trace_text, trace_flag=False):

    system_message = "You are a traffic scene analysis assistant. The user will provide the task description, reasoning steps, and necessary information. " \
            "You must strictly follow the reasoning steps specified by the user, analyze step by step, and output the results. " \
            "The reasoning process and the answer are contained in the <think> </think> and <answer> </answer> tags respectively, for example, <think> The reasoning process is here </think> <answer> The answer is here </answer>"

    if trace_flag:
        user_message = "<image>\n# Task:\nIdentify the traffic lights that the driver needs to pay attention to (hereinafter called attention light). Output their light status and infer the direction they indicate. " \
            "If there is a traffic light with a countdown number in the picture, output the remaining time of the attention light (if any). " \
            "Additionally, you must analyze the vehicle trajectory and its relationship with the traffic lights, including whether the vehicle is stopping, turning, or accelerating, and how its movement corresponds to the traffic light indications.\n" \
            "# You need to follow the solution step by step:\n" \
            "## Step 1: Determine the relationship between the scene and the intersection.\n" \
            "### Tips 1: The scene are divided into four categories: in front of a one-way intersection, in front of a normal/large intersection, on the road, passing the intersection.\n\n" \
            "## Step 2: Identify traffic lights that drivers should pay attention to.\n" \
            "### Tips 1: Identify the attributes of traffic lights in the image one by one, such as type, orientation, pattern, mounting position, etc.\n" \
            "### Tips 2: Analyze whether it needs to be paid attention to by the driver from the above attributes. Give ✅ and ❌ to distinguish the results, and give ⚠️ to those countdown lights.\n\n" \
            "## Step 3: Outputs the traffic light status, the direction it is indicating, and the countdown number (if any). \n" \
            "### Tips 1: From the traffic lights in step 2, identify the light state, and infer the direction indicated by combining the pattern and the light state.\n" \
            "### Tips 2: If there is a countdown light, analyze which traffic light the countdown numbers in the picture belong to.\n" \
            "## Step 4: Analyze the vehicle trajectory and its relationship with the traffic lights. \n" \
            "### Tips 1: Vehicle trajectory provides speed, direction, and heading change, which help confirm whether the vehicle is stopping, turning, or accelerating.\n" \
            "### Tips 2: If the vehicle is accelerating and the traffic light is green, it might be passing through the intersection. If decelerating with a red light, it might be stopping before the intersection.\n" \
            "### Tips 3: Consider if the light the vehicle is responding to is actually one of the attention traffic lights from Step 2.\n" \
            "### Tips 4: Use the following trajectory information for your reasoning:\n" \
            f"{summary_trace_text}\n\n" \
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
            "Below are the bounding boxes of the traffic lights in the picture one by one in order of sequence:\n" \
            f"{generate_traffic_light_references(round_bboxes(bbox_hint))}"
    else:
        user_message = "<image>\n# Task:\nIdentify the traffic lights that the driver needs to pay attention to (hereinafter called attention light). Output their light status and infer the direction they indicate. " \
            "If there is a traffic light with a countdown number in the picture, output the remaining time of the attention light (if any).\n" \
            "# You need to follow the solution step by step:\n" \
            "## Step 1: Determine the relationship between the scene and the intersection.\n" \
            "### Tips 1: The scene are divided into four categories: in front of a one-way intersection, in front of a normal/large intersection, on the road, passing the intersection.\n\n" \
            "## Step 2: Identify traffic lights that drivers should pay attention to.\n" \
            "### Tips 1: Identify the attributes of traffic lights in the image one by one, such as type, orientation, pattern, mounting position, etc.\n" \
            "### Tips 2: Analyze whether it needs to be paid attention to by the driver from the above attributes. Give ✅ and ❌ to distinguish the results, and give ⚠️ to those countdown lights.\n\n" \
            "## Step 3: Outputs the traffic light status, the direction it is indicating, and the countdown number (if any). \n" \
            "### Tips 1: From the traffic lights in step 2, identify the light state, and infer the direction indicated by combining the pattern and the light state.\n" \
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
            "Below are the bounding boxes of the traffic lights in the picture one by one in order of sequence:\n" \
            f"{generate_traffic_light_references(round_bboxes(bbox_hint))}"

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
                "content": cot
            }
        ],
        "images": [image_path]
    }

    return entry

def build_instruction_data_end2end(metainfo, cot, image_path, summary_trace_text=None, trace_flag=False):
    def generate_traffic_light_strings(numbers):
        return [f"Traffic Light {num + 1}" for num in numbers] 
    system_message = """You are a traffic scene analysis assistant. The user will provide the task description, reasoning steps, and necessary information. You must strictly follow the reasoning steps specified by the user, analyze step by step, and output the results.
The reasoning process and the answer are contained in the <think> </think> and <answer> </answer> tags respectively, for example <think>The reasoning process is here </think> <answer>The answer is here </answer>
"""

    if trace_flag and summary_trace_text:
        user_message = f"""<image># Task:
Identify the traffic lights that the driver needs to pay attention to. Output their light status and infer the direction they indicate. If there is a countdown light, bind the countdown number to the correct traffic light.
Additionally, you must analyze the vehicle trajectory and its relationship with the traffic lights, including whether the vehicle is stopping, turning, or accelerating, and how its movement corresponds to the traffic light indications.
Use the following trajectory information for your reasoning:
{summary_trace_text}

You need to follow the solution step by step:

## Step 1: Determine the relationship between the scene and the intersection.
## Step 2: Identify traffic lights that drivers should pay attention to.
## Step 3: Outputs the traffic light status, the direction it is indicating, and the countdown number (if any).
## Step 4: Analyze the vehicle trajectory and its relationship with the traffic lights.

# Output format:
<think>
    The reasoning process is here.
</think>
<answer>
[
    {{
        "bbox_2d": [x1,y1,x2,y2],
        "indications": ["left-turn"], 
        "state": "red",
        "countdown_number": None or Integer Number
    }},
    ...
]
</answer>
"""
    else:
        user_message = f"""<image># Task:
Identify the traffic lights that the driver needs to pay attention to. Output their light status and infer the direction they indicate. If there is a countdown light, bind the countdown number to the correct traffic light.

You need to follow the solution step by step:

## Step 1: Determine the relationship between the scene and the intersection.
## Step 2: Identify traffic lights that drivers should pay attention to.
## Step 3: Outputs the traffic light status, the direction it is indicating, and the countdown number (if any).

# Output format:
<think>
    The reasoning process is here.
</think>
<answer>
[
    {{
        "bbox_2d": [x1,y1,x2,y2],
        "indications": ["left-turn"], 
        "state": "red",
        "countdown_number": None or Integer Number
    }},
    ...
]
</answer>
"""

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
                "content": cot
            }
        ],
        "images": [image_path],
        "objects": {
            "ref": generate_traffic_light_strings([i for i in range(metainfo["total_num_of_traffic_lights"])] + [i-1 for i in metainfo["focusedTrafficLightIndex"]]),
            "bbox": metainfo["bboxes"] + [metainfo["bboxes"][i-1] for i in metainfo["focusedTrafficLightIndex"]]
        }
    }

    return entry

def build_instruction_data_noCoT(bbox_hint, image_path, answer, summary_trace_text=None, trace_flag=False):  
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
    system_message = """You are a traffic scene analysis assistant. """

    if trace_flag and summary_trace_text:
        user_message = f"""# Task:
Identify the traffic lights that the driver needs to pay attention to. Output their light status and infer the direction they indicate. If there is a countdown light, bind the countdown number to the correct traffic light.
Additionally, you must analyze the vehicle trajectory and its relationship with the traffic lights, including whether the vehicle is stopping, turning, or accelerating, and how its movement corresponds to the traffic light indications.
Use the following trajectory information for your reasoning:
{summary_trace_text}

# Output format:
<answer>
[
    {{
        "bbox_2d": [x1,y1,x2,y2],
        "indications": ["left-turn"], 
        "state": "red",
        "countdown_number": None or Integer Number
    }},
    ...
]
</answer>

# Below are the bounding boxes of the traffic lights in the picture one by one in order of sequence:
{round_bboxes(bbox_hint)}
"""
    else:
        user_message = f"""# Task:
Identify the traffic lights that the driver needs to pay attention to. Output their light status and infer the direction they indicate. If there is a countdown light, bind the countdown number to the correct traffic light.

# Output format:
<answer>
[
    {{
        "bbox_2d": [x1,y1,x2,y2],
        "indications": ["left-turn"], 
        "state": "red",
        "countdown_number": None or Integer Number
    }},
    ...
]
</answer>

# Below are the bounding boxes of the traffic lights in the picture one by one in order of sequence:
{round_bboxes(bbox_hint)}
"""

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
                "content": answer
            }
        ],
        "images": [image_path]
    }

    return entry

def build_instruction_data_noCoT_end2end(metainfo, answer, image_path, summary_trace_text=None, trace_flag=False):
    def generate_traffic_light_strings(numbers):
        return [f"Traffic Light {num + 1}" for num in numbers]
    system_message = """You are a traffic scene analysis assistant. """

    if trace_flag and summary_trace_text:
        user_message = f"""<image># Task:
Identify the traffic lights that the driver needs to pay attention to. Output their light status and infer the direction they indicate. If there is a countdown light, bind the countdown number to the correct traffic light.
Additionally, you must analyze the vehicle trajectory and its relationship with the traffic lights, including whether the vehicle is stopping, turning, or accelerating, and how its movement corresponds to the traffic light indications.
Use the following trajectory information for your reasoning:
{summary_trace_text}

# Output format:
<answer>
[
    {{
        "bbox_2d": [x1,y1,x2,y2],
        "indications": ["left-turn"], 
        "state": "red",
        "countdown_number": None or Integer Number
    }},
    ...
]
</answer>
"""
    else:
        user_message = f"""<image># Task:
Identify the traffic lights that the driver needs to pay attention to. Output their light status and infer the direction they indicate. If there is a countdown light, bind the countdown number to the correct traffic light.

# Output format:
<answer>
[
    {{
        "bbox_2d": [x1,y1,x2,y2],
        "indications": ["left-turn"], 
        "state": "red",
        "countdown_number": None or Integer Number
    }},
    ...
]
</answer>
"""

    entry = {
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": answer
            }
        ],
        "images": [image_path],
        "objects": {
            "ref": generate_traffic_light_strings([i for i in range(metainfo["total_num_of_traffic_lights"])] + [i-1 for i in metainfo["focusedTrafficLightIndex"]]),
            "bbox": metainfo["bboxes"] + [metainfo["bboxes"][i-1] for i in metainfo["focusedTrafficLightIndex"]]
        }
    }

    return entry