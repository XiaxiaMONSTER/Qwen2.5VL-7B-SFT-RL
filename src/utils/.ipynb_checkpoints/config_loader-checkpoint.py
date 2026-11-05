import os
import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """
    加载YAML配置文件
    :param config_path: 配置文件路径（相对或绝对）
    :return: 配置字典
    """
    try:
        # 获取项目根目录
        project_root = Path(__file__).resolve().parents[2]
        full_path = project_root / config_path
        
        with open(full_path, 'r') as file:
            config = yaml.safe_load(file)
        
            if 'API_KEY' in os.environ:
                config['api_key'] = os.environ['API_KEY']
                
            return config
    except FileNotFoundError:
        raise Exception(f"配置文件不存在: {full_path}")
    except yaml.YAMLError as e:
        raise Exception(f"配置文件解析失败: {str(e)}")