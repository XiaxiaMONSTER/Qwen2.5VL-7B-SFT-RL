import os
import base64

def image_to_base64(file_path):
    # 获取文件扩展名
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # 根据扩展名确定 MIME 类型
    if ext in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif ext in ['.png']:
        mime_type = 'image/png'
    elif ext in ['.gif']:
        mime_type = 'image/gif'
    else:
        raise ValueError(f"Unsupported image format: {ext}")

    # 读取并编码图像
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    return f"data:{mime_type};base64,{encoded_string}"