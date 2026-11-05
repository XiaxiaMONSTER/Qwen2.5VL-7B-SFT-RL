import json
from reward_function import FormatReward, BboxReward, StateReward, IndicationsReward, CountdownReward
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

from src.utils.assessment import match_answer

# output = "<think>\nOK, I need to identify the traffic light that the driver should focus on. Output the state of the light and infer the direction it indicates. If there is a countdown light in the light group, I should also confirm the countdown number associated with the traffic light. To accomplish this task, I need to break it down into following steps:\n# Step 1: Determine the relationship between the scene and the intersection.\nThe current scene is 'passing the intersection'. From the picture, we can see that the current vehicle may have crossed the stop line before the intersection and has not yet entered the next intersection, which means that the current vehicle is in the intersection. The traffic lights above the road are clearly visible. \n\n# Step 2: Identify traffic lights that drivers should pay attention to. \nThere are 11 traffic lights in the image. I will check the traffic lights one by one. And give ✅ to those traffic lights that drivers should pay attention to, ❌ to those traffic lights that drivers don't need to pay attention to, and ⚠️ to those with countdown.\n- Traffic light 1: This is a traffic light for motor vehicles at an intersection, attached to a horizontal bar. It is facing left and displaying an unidentifiable symbol. It is not a light that the driver needs to pay attention to. ❌\n- Traffic light 2: This is a traffic light for motor vehicles at an intersection, attached to a horizontal bar. It is facing away from the driver and displaying an unidentifiable symbol. It is not a light that the driver needs to pay attention to. ❌\n- Traffic light 3: This is a traffic light for motor vehicles at an intersection, mounted on vertical poles. It is facing the driver and displaying a Left arrow. It is the light that the driver needs to pay attention to. ✅\n- Traffic light 4: This is a traffic light for non-motor vehicles or pedestrians at an intersection, mounted on vertical poles. It is facing right and displaying an unidentifiable symbol. It is not a light that the driver needs to pay attention to. ❌\n- Traffic light 5: This is a traffic light for motor vehicles at an intersection, mounted on vertical poles. It is facing the driver and displaying a Left arrow. It is the light that the driver needs to pay attention to. ✅\n- Traffic light 6: This is a traffic light for motor vehicles at an intersection, attached to a horizontal bar. It is facing away from the driver and displaying an unidentifiable symbol. It is not a light that the driver needs to pay attention to. ❌\n- Traffic light 7: This is a traffic light for motor vehicles at an intersection, attached to a horizontal bar. It is facing left and displaying an unidentifiable symbol. It is not a light that the driver needs to pay attention to. ❌\n- Traffic light 8: This is a traffic light for motor vehicles at an intersection, attached to a horizontal bar. It is facing away from the driver and displaying an unidentifiable symbol. It is not a light that the driver needs to pay attention to. ❌\n- Traffic light 9: This is a traffic light for motor vehicles at an intersection, mounted on vertical poles. It is facing the driver and displaying an unidentifiable symbol. It is the light that the driver needs to pay attention to. ✅\n- Traffic light 10: This is a traffic light with a countdown timer, attached to a horizontal bar. It is facing away from the driver and displaying an unidentifiable symbol. It is not a light that the driver needs to pay attention to. ❌\n- Traffic light 11: This is a traffic light for motor vehicles at an intersection, mounted on vertical poles. It is facing away from the driver and displaying an unidentifiable symbol. It is not a light that the driver needs to pay attention to. ❌\n\n# Step 3:Outputs the traffic light status, the direction it is indicating, and the countdown number (if any). \nIn the second step, I found 3 traffic lights that drivers need to pay attention to. Next, I will analyze these 3 traffic lights one by one.\n- Traffic light 3: From the figure, we can see that its pattern is a green left arrow pattern. So the light state is green. Based on the pattern displayed by traffic lights, we can infer that the indicated direction is left-turn.\n- Traffic light 5: From the figure, we can see that its pattern is a green left arrow pattern. So the light state is green. Based on the pattern displayed by traffic lights, we can infer that the indicated direction is left-turn.\n- Traffic light 9: From the figure, we can see that its pattern is an unidentifiable unknown pattern. So the light state is unknown. It is not possible to infer the direction indicated by the traffic lights.\n</think>\n<answer>\n[{'bbox_2d': [993, 452, 1009, 492], 'indications': ['left-turn'], 'state': 'green', 'countdown_number': None}, {'bbox_2d': [968, 451, 982, 492], 'indications': ['left-turn'], 'state': 'green', 'countdown_number': None}, {'bbox_2d': [1019, 381, 1037, 429], 'indications': ['unknown'], 'state': 'unknown', 'countdown_number': None}]\n</answer>"
# answer = "<answer>\n[{'bbox_2d': [993, 452, 1009, 492], 'indications': ['left-turn'], 'state': 'green', 'countdown_number': None}, {'bbox_2d': [968, 451, 982, 492], 'indications': ['left-turn'], 'state': 'green', 'countdown_number': None}]\n</answer>"

output_json_path = r"/root/autodl-tmp/traffic_lights/results/model_output/QwenCoT.json"
eval_path = r"/root/autodl-tmp/traffic_lights/data/instructions/eval_dataset_cot_template.json"
test_index = 199

with open(output_json_path, 'r') as f:
    model_result = json.load(f)["output"]

with open(eval_path, 'r') as f:
    answer_response = json.load(f)

item = model_result[test_index]
image_path = item["image_path"][0]

index = None
for i, resp in enumerate(answer_response):
    if resp["images"][0] == image_path:
        index = i
        break

model_response = item["response"]
# model_answer = match_answer(item["response"])
really_answer = match_answer(answer_response[index]["messages"][2]["content"])['traffic_light']
answer_text = f"<answer>{really_answer}</answer>"
print("模型回复:\n", model_response)
print("真实答案:\n", answer_text)



# output = model_result[index]["response"]

format_reward = FormatReward()
state_reward = StateReward()
bbox_reward = BboxReward()
# bbox_reward_end2end = BboxReward(is_end_to_end=True)
indications_reward = IndicationsReward()
countdown_reward = CountdownReward()


print("格式奖励", format_reward([model_response]))
print("灯态奖励", state_reward([model_response],[answer_text]))
print("bbox奖励", bbox_reward([model_response],[answer_text]))
print("指示奖励", indications_reward([model_response],[answer_text]))
print("倒计时奖励", countdown_reward([model_response],[answer_text]))
