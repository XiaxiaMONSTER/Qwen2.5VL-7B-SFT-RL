import json

with open('./data/instructions/marked-trace-smart_resize-cot_template-42.json', 'r') as f:
    data = json.load(f)
print('总数据条数:', len(data))