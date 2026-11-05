import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer
# model_path = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
model_path = r"/root/autodl-tmp/traffic_lights/models/Qwen2.5VL-CoT_Template/v0-20250720-132016/checkpoint-600"
assert os.path.exists(model_path)

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )


# processor = AutoProcessor.from_pretrained(
#     model_path,
# )

print('\n模型加载完毕！')

# 实验部分

PROMPT = """<image> # Task: Identify the traffic lights that the driver needs to pay attention to (hereinafter called attention light). Output their light status and infer the direction they indicate. If there is a traffic light with a countdown number in the picture, output the remaining time of the attention light (if any). # You need to follow the solution step by step: ## Step 1: Determine the relationship between the scene and the intersection. ### Tips 1: The scene are divided into four categories: in front of a one-way intersection, in front of a normal/large intersection, on the road, passing the intersection. ## Step 2: Identify traffic lights that drivers should pay attention to. ### Tips 1: Identify the attributes of traffic lights in the image one by one, such as type, orientation, pattern, mounting position, etc. ### Tips 2: Analyze whether it needs to be paid attention to by the driver from the above attributes. Give ✅ and ❌ to distinguish the results, and give ⚠️ to those countdown lights. ## Step 3: Outputs the traffic light status, the direction it is indicating, and the countdown number (if any). ### Tips 1: From the traffic lights in step 2, identify the light state, and infer the direction indicated by combining the pattern and the light state. ### Tips 2: If there is a countdown light, analyze which traffic light the countdown numbers in the picture belong to. # Output format: <think> The reasoning process is here. </think> <answer> [ {"bbox_2d": [x1,y1,x2,y2], "indications": ["left-turn"], "state":"green", "countdown_number": None or Integer Number} {"bbox_2d": [x1,y1,x2,y2], "indications": ["U-turn"], "state":"red", "countdown_number": None or Integer Number} ] </answer> Below are the bounding boxes of the traffic lights in the picture one by one in order of sequence: <|object_ref_start|>Traffic Light 1<|object_ref_end|>: <|box_start|>[1103, 696, 1114, 708]<|box_end|> <|object_ref_start|>Traffic Light 2<|object_ref_end|>: <|box_start|>[1137, 762, 1144, 770]<|box_end|> <|object_ref_start|>Traffic Light 3<|object_ref_end|>: <|box_start|>[1163, 679, 1177, 708]<|box_end|> <|object_ref_start|>Traffic Light 4<|object_ref_end|>: <|box_start|>[1027, 698, 1038, 710]<|box_end|> <|object_ref_start|>Traffic Light 5<|object_ref_end|>: <|box_start|>[967, 699, 980, 712]<|box_end|> <|object_ref_start|>Traffic Light 6<|object_ref_end|>: <|box_start|>[1230, 676, 1241, 704]<|box_end|>"""

image_path = r"./data/processed/marked-smart_resize-42/image_07314.jpg"

        
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # 图片可以是 URL、本地图片路径、base64编码（data:image;base64......）
                "image": image_path,
            },
            {"type": "text", "text": PROMPT},
        ],
    }
]


 
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)
 
text = '<image> # Task: Identify the traffic lights that the driver needs to pay attention to (hereinafter called attention light). Output their light status and infer the direction they indicate. If there is a traffic light with a countdown number in the picture, output the remaining time of the attention light (if any). # You need to follow the solution step by step: ## Step 1: Determine the relationship between the scene and the intersection. ### Tips 1: The scene are divided into four categories: in front of a one-way intersection, in front of a normal/large intersection, on the road, passing the intersection. ## Step 2: Identify traffic lights that drivers should pay attention to. ### Tips 1: Identify the attributes of traffic lights in the image one by one, such as type, orientation, pattern, mounting position, etc. ### Tips 2: Analyze whether it needs to be paid attention to by the driver from the above attributes. Give ✅ and ❌ to distinguish the results, and give ⚠️ to those countdown lights. ## Step 3: Outputs the traffic light status, the direction it is indicating, and the countdown number (if any). ### Tips 1: From the traffic lights in step 2, identify the light state, and infer the direction indicated by combining the pattern and the light state. ### Tips 2: If there is a countdown light, analyze which traffic light the countdown numbers in the picture belong to. # Output format: <think> The reasoning process is here. </think> <answer> [ {"bbox_2d": [x1,y1,x2,y2], "indications": ["left-turn"], "state":"green", "countdown_number": None or Integer Number} {"bbox_2d": [x1,y1,x2,y2], "indications": ["U-turn"], "state":"red", "countdown_number": None or Integer Number} ] </answer> Below are the bounding boxes of the traffic lights in the picture one by one in order of sequence: <|object_ref_start|>Traffic Light 1<|object_ref_end|> : <|box_start|>[1105, 692, 1117, 704]<|box_end|> <|object_ref_start|>Traffic Light 2<|object_ref_end|> : <|box_start|>[1235, 670, 1246, 701]<|box_end|> <|object_ref_start|>Traffic Light 3<|object_ref_end|> : <|box_start|>[1169, 675, 1180, 704]<|box_end|> <|object_ref_start|>Traffic Light 4<|object_ref_end|> : <|box_start|>[1027, 694, 1040, 707]<|box_end|> <|object_ref_start|>Traffic Light 5<|object_ref_end|> : <|box_start|>[965, 696, 979, 708]<|box_end|>'
tokens = tokenizer.tokenize(text)
 
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
 
print(ids)
 




# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to(model.device)

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=4096)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )


# print(output_text)