import json
import os
from utils.api_client import chat
from utils.annotation_processing import translate_annotations_zh_to_en, generate_metainfo
from utils.config_loader import load_config
def main():
    config = load_config()
    annotations_file_folder = config["data"]["annotations_folder"]
    # annotations_file_folder = r"./data/annotations/"
    high_quality_path = "/root/autodl-tmp/pure_code/data/json/high_quality_annotations.json"
    with open(high_quality_path, 'r') as f:
        high_quality = json.load(f)

    for file_name,info in high_quality["details"].items():
        descriptions = []
        base_name = os.path.splitext(file_name)[0]
        json_file_path = os.path.join(annotations_file_folder, file_name)
        info = ""
        print(base_name)
        try:
            # 读取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 处理数据
            annotations_en = translate_annotations_zh_to_en(annotations_zh=data)
            metainfo = generate_metainfo(annotation_en=annotations_en)
            print(metainfo)
            img_path = os.path.join(config["data"]["processed_image_folder"], "marked-smart_resize-cot_template-42",f"{base_name}.jpg")
            for index in range(metainfo["total_num_of_traffic_lights"]):
                info += f"Traffic Light {index+1}: "
                if int(index+1) in metainfo["focusedTrafficLightIndex"]:
                    info += "need to pay attention, "
                else:
                    info += "no need to pay attention, "
                info += f"{metainfo["type"][index]}, "
                pictogram = "+".join([item for item in metainfo["pictogram"][index] if item != 'NA'])
                info += f"{pictogram}, "
                info += f"{metainfo["direction"][index]}, "
                info += f"{metainfo["light_state"][index]}, "
                info += f"{metainfo["attachment"][index]}\n"

        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            continue  # 跳过出错的文件
        
        try:
            prompt = """I will provide the type, pattern, physical orientation (Its physical orientation toward the front means it faces the driver), light status , attachments and whether the driver needs to pay attention (This means drivers should look at this light when passing through the intersection). You need to describe it based on these attributes and the actual situation presented in the picture. When describing, please add a description of the light group (the same attachments are considered a group) and location (if it is clear), such as the coordinates of a certain traffic light, and it is in the same light group as a countdown light. Infer the reason why this light needs attention.
            The answer format is:
            "Traffic Light X: A paragraph describing the details of a traffic light X. The reason why this light is being focused or not focused. It is (or not) the light that the driver needs to pay attention to. "
            Here is the information I provided:
            {}
            """.format(info)

            print(info)
            model_name = "qvq-max"
            answer_content = chat(image_path=img_path, prompt=prompt)
            print(answer_content)        
            descriptions.append({"image_path":img_path, "description":answer_content, "model_name":model_name})
            break
        except Exception as e:
            print(f"AI处理 {file_name} 时出错: {str(e)}")
            break
        

    with open("./results/traffic_light_description.json", 'w') as  f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

