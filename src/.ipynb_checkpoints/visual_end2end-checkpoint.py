import json
import os
import torch
import pandas as pd
import re
from ast import literal_eval
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from PIL import Image
from utils.assessment import match_answer
from utils.visualization import visual_model_response
from utils.annotation_processing import translate_annotations_zh_to_en, generate_metainfo
from qwen_vl_utils import smart_resize



def main2():
    model_path = r""
    val_path = r""
    model_output_path = r""
    index = 0


    with open(val_path, 'r') as f:
        data = json.load(f)

    assert os.path.exists(model_path)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        attn_implementation="flash_attention_2",
        )
    processor = AutoProcessor.from_pretrained(
        model_path,
        )

    item = data[index]
    image_path = item["images"]
    system_prompt = item["messages"][0]["content"]
    prompt = item["messages"][1]["content"]
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)


    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    model_dict = match_answer(output_text[0])
    print(model_dict)




    img = Image.open(image_path)
    w, h = img.size
    new_h, new_w = smart_resize(height=h, width=w)
    width_ratio = new_w / w
    height_ratio = new_h / h

    annotations_en = translate_annotations_zh_to_en(annotations_zh=ann)
    answer_metainfo = generate_metainfo(
        annotation_en=annotations_en, 
        scaling_factor=(width_ratio, height_ratio)
    )

    visual_model_response(
        image_path=image_path, 
        answer_dict=model_dict, 
        output_path=model_output_path,
        answer_metainfo=answer_metainfo,
        target_size=(new_w, new_h)
    )


def main():
    from utils.tools import inference, plot_bounding_boxes
    image_path = "./assets/spatial_understanding/cakes.png"


    ## Use a local HuggingFace model to inference.
    # prompt in chinese
    prompt = "框出每一个小蛋糕的位置，以json格式输出所有的坐标"
    # prompt in english
    prompt = "Outline the position of each small cake and output all the coordinates in JSON format."
    response, input_height, input_width = inference(image_path, prompt)

    image = Image.open(image_path)
    print(image.size)
    image.thumbnail([640,640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image,response,input_width,input_height)

if __name__ == "__main__":
    main()