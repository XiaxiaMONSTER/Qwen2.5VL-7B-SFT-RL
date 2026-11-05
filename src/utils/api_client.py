from openai import OpenAI
import asyncio
from openai import AsyncOpenAI
from utils.config_loader import load_config
from utils.images_processing import image_to_base64

def chat(prompt, image_path, model_name="qwen-vl-max-latest"):
    config = load_config()
    answer_content = "" 
    local_image_b64 = image_to_base64(image_path)

    client = OpenAI(
            # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
            api_key = config["api_key"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    # 创建聊天完成请求
    completion = client.chat.completions.create(
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
        stream=True,
        # 解除以下注释会在最后一个chunk返回Token使用量
        # stream_options={
        #     "include_usage": True
        # }
    )
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            continue
        if delta.content:
            answer_content += delta.content
    return answer_content

async def async_chat(prompt, image_path, model_name="qwen-vl-max-latest"):
    """
    异步版本的聊天函数，使用OpenAI的异步API
    """
    config = load_config()
    answer_content = ""
    local_image_b64 = image_to_base64(image_path)
    
    # 创建异步客户端
    client = AsyncOpenAI(
        api_key=config["api_key"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
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
                
    except Exception as e:
        # 捕获并重新抛出异常，包含模型名称信息
        raise Exception(f"模型 {model_name} 调用失败: {str(e)}")
    
    finally:
        # 确保关闭客户端连接
        await client.close()
        
    return answer_content