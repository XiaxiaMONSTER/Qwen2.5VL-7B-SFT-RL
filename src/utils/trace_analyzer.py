import numpy as np
import pandas as pd
import os

def analyze_traj_with_light(row: dict, light_info: dict) -> str:
    # 提取轨迹信息
    tr_state = row.get("state", "")
    tr_turn = row.get("turn_label", "")
    hdg = row.get("turn_angle_deg", 0)
    drop = row.get("delta_v", 0)

    # 提取灯态信息
    state = str(light_info.get("light_state", "")).lower()
    indication = light_info.get("indication", [])
    ltype = light_info.get("pictogram", "")

    # 开始分析
    rules = []

    # 灯态影响
    if state in ["yellow"] and tr_state == "减速":
        rules.append("The vehicle decelerated under a yellow light, likely in response to the signal.")
    if state in ["red"] and tr_state == "减速":
        rules.append("The vehicle slowed down before a red light, likely preparing to stop.")
    if state in ["red"] and tr_state == "静止":
        rules.append("The vehicle was stationary at a red light, likely waiting for green.")
    if state in ["green"] and tr_state in ["加速", "匀速"]:
        rules.append("The vehicle maintained speed or accelerated under a green light.")

    # 方向一致性
    if "左" in indication and tr_turn == "左转":
        rules.append("The vehicle made a left turn, consistent with the left-turn signal.")
    if "右" in indication and tr_turn == "右转":
        rules.append("The vehicle made a right turn, consistent with the right-turn signal.")
    if "直行" in indication and tr_turn == "直行":
        rules.append("The vehicle proceeded straight, consistent with the straight signal.")

    # 兜底逻辑
    if not rules:
        rules.append("No clear trajectory change was found to be directly related to the light state.")

    return "Trajectory-Light Relationship:\n" + " ".join(rules)

def read_txt(txt_path: str) -> pd.DataFrame:
    # 不让 pandas 用第一行做列名，而是直接给默认列名 0,1,2...
    df = pd.read_csv(txt_path, header=None)
    # 按列号重命名，方便后续计算
    df.columns = [
        'seq','time','exp_time',
        'enu_x','enu_y','enu_z',
        'enu_rx','enu_ry','enu_rz',
        'enu_vx','enu_vy','enu_vz',
        'fix_type'
    ]
    # 转类型
    df['seq'] = pd.to_numeric(df['seq'], errors='coerce').astype('Int64')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['enu_vx'] = pd.to_numeric(df['enu_vx'], errors='coerce')
    df['enu_vy'] = pd.to_numeric(df['enu_vy'], errors='coerce')
    df['enu_vz'] = pd.to_numeric(df['enu_vz'], errors='coerce')
    df['enu_rz'] = pd.to_numeric(df['enu_rz'], errors='coerce')
    df['enu_x'] = pd.to_numeric(df['enu_x'], errors='coerce')
    df['enu_y'] = pd.to_numeric(df['enu_y'], errors='coerce')
    return df

def analyze_vehicle_behavior_from_trajectory(traj_path, start_seq, end_seq, light_info):
    """
    根据轨迹数据分析车辆行为，并结合红绿灯状态判断行为是否合理。

    参数：
        traj_path (str): 轨迹 .txt 文件路径
        start_seq (int): 分析的起始帧编号
        end_seq (int): 分析的结束帧编号
        light_info (dict): 包含当前关注灯的颜色、图案、指向等信息

    返回：
        str: 可拼接入 CoT 的自然语言描述
    """

    df = read_txt(traj_path)
    # === Step 2: 提取分析区间的数据 ===
    df_window = df[(df['seq'] >= start_seq) & (df['seq'] <= end_seq)].reset_index(drop=True)

    if df_window.empty:
        return f"No valid trajectory points between seq {start_seq} and {end_seq}."

    if len(df_window) < 2:
        return f"Trajectory segment [{start_seq}, {end_seq}] is too short to analyze vehicle behavior."

    start = df_window.iloc[0]
    end = df_window.iloc[-1]

    # === Step 3: 分析车辆状态 ===
    try:
        start_v = np.sqrt(start['enu_vx'] ** 2 + start['enu_vy'] ** 2)
        end_v = np.sqrt(end['enu_vx'] ** 2 + end['enu_vy'] ** 2)
        delta_v = end_v - start_v

        delta_heading = end['enu_rz'] - start['enu_rz']
        delta_heading_deg = abs(delta_heading) * 180 / np.pi
    except Exception as e:
        # 打印
        return f"Failed to calculate speed or heading change: {e}"

    # === Step 4: 获取红绿灯信息 ===
    color = light_info.get("light_state", "unknown")
    pictogram = light_info.get("pictogram", [])
    direction = light_info.get("indication", [])
    is_attention = light_info.get("is_attention", False)

    # === Step 5: 生成行为推理描述 ===
    desc = []

    # 速度分析
    if abs(delta_v) > 0.5:
        if delta_v > 0:
            desc.append("The vehicle is accelerating.")
            if color == "green" and is_attention:
                desc.append("Since the traffic light is green and a focus light, the acceleration is reasonable.")
            elif color == "red" and is_attention:
                desc.append("However, the traffic light is red and a focus light, which may indicate risky behavior.")
        else:
            desc.append("The vehicle is decelerating.")
            if color == "red" and is_attention:
                desc.append("Since the red light is focused, this deceleration is appropriate.")
            elif color == "green" and is_attention:
                desc.append("Decelerating under green light may suggest obstacle or hesitation.")

    # 方向分析
    if delta_heading_deg > 15:
        turn = "left" if delta_heading > 0 else "right"
        desc.append(f"The vehicle is turning {turn}.")
        if is_attention and color == "green":
            if turn == "left" and ("left" in direction or "straight" in direction) and (
                "arrow" in "".join(pictogram) or "circle" in pictogram):
                desc.append("The green light allows a left turn with valid pattern and direction.")
            elif turn == "right":
                desc.append("The green light may allow a right turn, but confirmation depends on traffic rules.")

    # 无显著行为
    if not desc:
        desc.append("No significant acceleration, deceleration, or turning behavior was detected.")

    # ✅ 最终结果拼接
    return " ".join(desc)
