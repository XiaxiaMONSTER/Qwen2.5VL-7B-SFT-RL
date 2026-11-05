from utils.search import search
from utils.annotation_processing import generate_annswer, generate_metainfo, translate_annotations_zh_to_en
from utils.generate_CoT import generate_CoT_by_template
import json
import os
folder_path = r"./data/annotations"
result = search(
    folder_path,
    {"scene": "普通/大型路口前"},
    require_pictogram_digit=True,
    n=11
)
# print(search(folder_path, query_dict))
print(result)
file = os.path.join(folder_path, result)
print(file)
with open(file,'r') as f:
    data = json.load(f)
# print(data,"\n\n")
annotations_en = translate_annotations_zh_to_en(annotations_zh=data)
# print(annotations_en)
metainfo = generate_metainfo(annotation_en=annotations_en)
print(generate_CoT_by_template(metainfo))
print(metainfo)

from eval_ann import evaluate_annotation_quality
print(evaluate_annotation_quality(metainfo=metainfo))