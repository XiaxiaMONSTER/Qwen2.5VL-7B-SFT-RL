import json
import os
import random
import asyncio
import time
from tqdm import tqdm
from openai import AsyncOpenAI
from utils.annotation_processing import translate_annotations_zh_to_en, generate_metainfo
from utils.config_loader import load_config
from utils.images_processing import image_to_base64

# 全局变量
MAX_CONCURRENT_REQUESTS = 10  # 控制并发请求数量
RETRY_DELAY = 5  # 重试延迟（秒）
MAX_RETRIES = 3  # 最大重试次数

# 异步聊天函数（增加重试机制）
async def async_chat(prompt, image_path, model_name="qwen-vl-max-latest", retry_count=0):
    """
    异步版本的聊天函数，带重试机制
    """
    if retry_count >= MAX_RETRIES:
        raise Exception(f"模型 {model_name} 调用失败: 超过最大重试次数")
    
    config = load_config()
    answer_content = ""
    local_image_b64 = image_to_base64(image_path)
    
    # 创建异步客户端
    client = AsyncOpenAI(
        api_key=config["api_key"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=120.0  # 设置超时时间
    )
    
    try:
        # 创建异步聊天完成请求
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": local_image_b64
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            stream=True
        )
        
        # 异步处理响应流
        async for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                continue
            if delta.content:
                answer_content += delta.content
                
        return answer_content
        
    except asyncio.TimeoutError:
        # 超时错误，等待后重试
        await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
        return await async_chat(prompt, image_path, model_name, retry_count + 1)
        
    except Exception as e:
        # 捕获并重新抛出异常，包含模型名称信息
        raise Exception(f"模型 {model_name} 调用失败: {str(e)}")
    
    finally:
        # 确保关闭客户端连接
        await client.close()

async def process_file(file_name, config, existing_descriptions, existing_base_names, 
                      result_file, available_models, progress_bar, semaphore):
    async with semaphore:  # 使用信号量控制并发
        base_name = os.path.splitext(file_name)[0]
        json_file_path = os.path.join(config["data"]["annotations_folder"], file_name)
        
        progress_bar.set_description(f"预处理: {base_name}")
        
        try:
            # 读取和处理JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            annotations_en = translate_annotations_zh_to_en(annotations_zh=data)
            metainfo = generate_metainfo(annotation_en=annotations_en)
            img_path = os.path.join(config["data"]["images_folder"],
                                   f"{base_name}.jpg")
            scene = metainfo["scene"]
            # 确保图片文件存在
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图片文件不存在: {img_path}")
            
        except Exception as e:
            progress_bar.set_description(f"错误: {base_name}")
            print(f"\n处理文件 {file_name} 时出错: {str(e)}")
            return
        
        prompt = f"""Your task is to explain the scene labels. The scenes can be roughly divided into: before the intersection, crossing the intersection, and outside the intersection (away from the intersection), and there are further subdivisions based on this. The classification is based on the position relationship between the vehicle and the intersection, the characteristics of the intersection, and whether the traffic lights at the intersection can be seen. Please use a paragraph to explain why the scene of this picture is "{scene}" based on the road elements in the picture (such as the number of lanes, traffic signs, traffic lights, intersection width, whether the pedestrian line is crossed, etc.). Start with 'The scene of this picture is "{scene}". ' The language is concise, no more than 150 words."""
        # 尝试使用不同模型直到成功或没有可用模型
        success = False
        answer_content = ""
        model_used = ""
        
        # 创建当前任务的模型副本，避免并发修改问题
        local_models = available_models.copy()
        
        while local_models and not success:
            model_name = random.choice(local_models)
            progress_bar.set_description(f"尝试 {model_name}: {base_name}")
            
            try:
                answer_content = await async_chat(
                    prompt=prompt, 
                    image_path=img_path, 
                    model_name=model_name
                )
                success = True
                model_used = model_name
                progress_bar.set_description(f"成功: {base_name}")
            except Exception as e:
                progress_bar.set_description(f"失败: {base_name}")
                print(f"\n模型 {model_name} 处理 {base_name} 时出错: {str(e)}")
                # 从本地副本中移除失败的模型
                local_models.remove(model_name)
                
                # 更新全局模型列表（使用锁确保线程安全）
                if model_name in available_models:
                    async with available_models_lock:
                        if model_name in available_models:
                            available_models.remove(model_name)
                            print(f"已从可用模型中移除: {model_name}")
                
                if not local_models:
                    print(f"\n文件 {base_name} 处理失败，所有模型均不可用")
                    return
        
        if not success:
            print(f"\n文件 {base_name} 处理失败，所有模型均不可用")
            return
        
        # 创建新条目
        new_entry = {
            "base_name": base_name,
            "image_path": img_path,
            "scene": scene,
            "description": answer_content, 
            "model_name": model_used
        }
        
        # 添加到现有结果
        existing_descriptions.append(new_entry)
        
        # 保存结果到文件
        with open(result_file, 'w') as f:
            json.dump(existing_descriptions, f, ensure_ascii=False, indent=2)
        
        # 更新已处理集合
        existing_base_names.add(base_name)
        progress_bar.update(1)

async def main():
    config = load_config()
    high_quality_path = "/root/autodl-tmp/pure_code/data/json/high_quality_annotations.json"
    result_file = "./results/scene_description.json"
    available_models = ["qwen-vl-max-2025-04-08", "qwen-vl-max-2025-01-25", "qwen-vl-max-2025-04-02", "qwen-vl-max-1030", "qwen-vl-max-1119", "qwen-vl-max-1230", "qwen-vl-max-0809"]  # 更新模型列表
    # 创建结果文件（如果不存在）
    if not os.path.exists(result_file):
        with open(result_file, 'w') as f:
            json.dump([], f)
    
    # 读取已有结果
    with open(result_file, 'r') as f:
        existing_descriptions = json.load(f)
    
    # 创建已处理base_name集合
    existing_base_names = {item["base_name"] for item in existing_descriptions}
    
    with open(high_quality_path, 'r') as f:
        high_quality = json.load(f)
    
    # 准备待处理文件列表
    files_to_process = []
    for file_name in high_quality["details"]:
        base_name = os.path.splitext(file_name)[0]
        if base_name not in existing_base_names:
            files_to_process.append(file_name)
    
    print(f"总文件数: {len(high_quality['details'])}")
    print(f"已处理文件: {len(existing_base_names)}")
    print(f"待处理文件: {len(files_to_process)}")
    
    if not files_to_process:
        print("所有文件已处理完成，无需进一步处理。")
        return
    
    # 创建进度条
    progress_bar = tqdm(total=len(files_to_process), desc="处理文件", unit="file")
    
    # 创建信号量控制并发量
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # 创建模型列表锁
    global available_models_lock
    available_models_lock = asyncio.Lock()
    
    # 创建并发任务
    tasks = []
    for file_name in files_to_process:
        # 如果所有模型都不可用了，停止创建新任务
        if not available_models:
            print("\n所有模型均不可用，停止创建新任务")
            break
            
        task = asyncio.create_task(
            process_file(
                file_name, config, existing_descriptions, 
                existing_base_names, result_file, available_models, progress_bar, semaphore
            )
        )
        tasks.append(task)
    
    # 等待所有任务完成
    await asyncio.gather(*tasks)
    
    progress_bar.close()
    print("\n文件处理完成！")
    print(f"剩余可用模型: {available_models}")

if __name__ == "__main__":
    asyncio.run(main())