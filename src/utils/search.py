import os
import json

import re

def contains_key_value(obj, target_key, target_value, require_pictogram_digit=False):
    """
    递归检查 obj 中是否存在键为 target_key 且值为 target_value 的项，
    同时可选检查是否包含带数字的 pictogram 属性。
    支持嵌套的 dict 和 list 结构。
    
    参数:
        require_pictogram_digit (bool): 是否要求存在带数字的 pictogram
    """
    found = False
    digit_in_pictogram = not require_pictogram_digit  # 如果不要求则默认为满足条件
    
    if isinstance(obj, dict):
        # 检查当前层级的键值对
        if target_key in obj:
            if obj[target_key] == target_value:
                found = True
                
        # 检查 pictogram 属性
        if require_pictogram_digit and not digit_in_pictogram:
            if "pictogram" in obj and isinstance(obj["pictogram"], list):
                for item in obj["pictogram"]:
                    if isinstance(item, str) and re.search(r'\d+', item):
                        digit_in_pictogram = True
                        break
        
        # 递归检查值
        for k, v in obj.items():
            child_found, child_digit = contains_key_value(
                v, target_key, target_value, require_pictogram_digit
            )
            found = found or child_found
            digit_in_pictogram = digit_in_pictogram or child_digit
            
    elif isinstance(obj, list):
        for item in obj:
            child_found, child_digit = contains_key_value(
                item, target_key, target_value, require_pictogram_digit
            )
            found = found or child_found
            digit_in_pictogram = digit_in_pictogram or child_digit
    
    return found, digit_in_pictogram


def search(folder_path, query_dict, require_pictogram_digit=False, n=1):
    """
    在指定文件夹中查找满足多个键值对的 JSON 标注文件，支持返回第 n 个匹配项
    
    参数:
        folder_path (str): 标注文件所在的文件夹路径
        query_dict (dict): 要搜索的键值对，如 {"scene": "普通/大型路口前", "state": "off"}
        require_pictogram_digit (bool): 是否要求存在带数字的 pictogram 属性
        n (int): 返回第 n 个匹配项（从 1 开始计数）
    
    返回:
        str: 第 n 个匹配的文件名（不含路径），若未找到则返回 None
    """
    if not query_dict:
        return None
    
    matches = []  # 存储所有匹配的文件名
    n = max(1, n)  # 确保 n 至少为 1
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".json"):
            continue
            
        file_path = os.path.join(folder_path, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"跳过文件 {filename}: {e}")
            continue
        
        # 检查所有查询条件
        all_match = True
        for key, value in query_dict.items():
            found, _ = contains_key_value(data, key, value, require_pictogram_digit)
            if not found:
                all_match = False
                break
        
        # 检查 pictogram 数字要求
        if all_match and require_pictogram_digit:
            _, digit_found = contains_key_value(data, "", "", True)
            if not digit_found:
                all_match = False
        
        if all_match:
            matches.append(filename)
            if len(matches) == n:
                return filename  # 找到第 n 个立即返回
    
    # 检查是否有足够的匹配项
    if len(matches) >= n:
        return matches[n-1]
    return None  # 未找到足够匹配项