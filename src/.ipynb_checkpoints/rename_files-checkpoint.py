import os
import argparse
import json
import shutil
from pathlib import Path

from utils.config_loader import load_config

def main(args):
    config = load_config()
    
    # 定义输入和输出路径
    ann_dir = Path(config["paths"]["raw_annotations"])
    img_dir = Path(config["paths"]["raw_dataset"])
    output_ann_dir = Path(config["data"]["annotations_folder"])
    output_img_dir = Path(config["data"]["images_folder"])
    mapping_file = Path(args.mapping_file)
    
    prefix = args.prefix
    dry_run = args.dry_run

    # 创建输出目录
    output_ann_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    mapping_file.parent.mkdir(parents=True, exist_ok=True)

    # 收集匹配的文件对
    matches = []
    for ann_file in ann_dir.glob("*.json"):
        base_name = ann_file.stem
        img_file = img_dir / f"{base_name}.jpg"
        if img_file.exists():
            matches.append((ann_file, img_file, base_name))
        else:
            print(f"警告：找不到对应的图片文件 {img_file}")

    # 检查是否有匹配项
    if not matches:
        print("未找到匹配的文件对，退出。")
        return

    # 按原文件名排序
    matches.sort(key=lambda x: x[0].name)

    # 开始处理文件
    print(f"将处理 {len(matches)} 对文件（前缀: {prefix}）：")
    name_mapping = {}
    
    for i, (ann_path, img_path, base_name) in enumerate(matches, start=1):
        new_name = f"{prefix}_{i:05d}"
        new_ann_path = output_ann_dir / f"{new_name}.json"
        new_img_path = output_img_dir / f"{new_name}.jpg"

        # 记录映射关系
        name_mapping[new_name] = base_name

        # 输出处理信息
        print(f"{base_name} → {new_name}")

        # 实际复制文件
        if not dry_run:
            # 复制标注文件
            shutil.copy(ann_path, new_ann_path)
            # 复制图片文件
            shutil.copy(img_path, new_img_path)

    # 保存映射文件
    if not dry_run:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(name_mapping, f, indent=4, ensure_ascii=False)
        print(f"映射文件已保存至：{mapping_file}")
    
    if dry_run:
        print("\n[干运行模式]：未实际复制文件，如需执行操作，请移除 --dry-run 参数。")
    else:
        print("\n文件复制和重命名完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一重命名标注文件和图片文件（仅处理有标注的图片）")
    parser.add_argument("--mapping-file", type=str, default="./data/json/name_mapping.json", help="名称映射JSON文件路径")
    parser.add_argument("--prefix", type=str, default="image", help="新文件名前缀(默认:image)")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际修改文件")
    args = parser.parse_args()
    main(args)