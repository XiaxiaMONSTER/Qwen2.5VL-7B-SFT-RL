import os
import glob
import numpy as np
import pandas as pd
import json

# ========== 参数设置 ==========
input_dir = "/root/autodl-tmp/dl0805/imgpost"   # 存放轨迹txt的目录
output_dir = "/root/autodl-tmp/traffic_lights/yanzheng/anal_results1"
os.makedirs(output_dir, exist_ok=True)

# 阈值
eps_speed = 0.2        # 静止阈值 (m/s)
acc_th = 0.05          # 加减速阈值 (m/s)
turn_th = np.radians(15)  # 转弯角度阈值 (弧度，15°)
min_segment_len = 10   # 最小区间长度（帧），避免碎片化

# ========== 读取数据 ==========
def read_txt(txt_path: str) -> pd.DataFrame:
    df = pd.read_csv(txt_path, header=None)
    df.columns = [
        'seq','time','exp_time',
        'enu_x','enu_y','enu_z',
        'enu_rx','enu_ry','enu_rz',
        'enu_vx','enu_vy','enu_vz',
        'fix_type'
    ]
    numeric_cols = ['seq','time','enu_x','enu_y','enu_z','enu_rz','enu_vx','enu_vy','enu_vz']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=["seq"]).reset_index(drop=True)
    df["seq"] = df["seq"].astype(int)
    return df

# ========== 辅助函数 ==========
def normalize_angle(angle):
    """将角度标准化到 [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_motion_state(speed, delta_v):
    """根据速度和加速度判定运动状态"""
    if speed < eps_speed:
        return "静止"
    elif delta_v > acc_th:
        return "加速"
    elif delta_v < -acc_th:
        return "减速"
    else:
        return "匀速"

# ========== 区间总结 ==========
def summarize_segment(df, start_idx, end_idx, state, start_heading):
    try:
        start_seq = int(df.iloc[start_idx]["seq"])
        end_seq   = int(df.iloc[end_idx]["seq"])
    except Exception:
        return None

    # 时间
    start_time = df.iloc[start_idx]["time"]
    end_time   = df.iloc[end_idx]["time"]
    duration   = end_time - start_time if pd.notna(end_time) and pd.notna(start_time) else np.nan

    # 速度统计
    speeds = df.loc[start_idx:end_idx, "speed_smooth"]
    start_speed = speeds.iloc[0]
    end_speed   = speeds.iloc[-1]
    avg_speed   = speeds.mean()
    max_speed   = speeds.max()
    min_speed   = speeds.min()
    delta_v     = end_speed - start_speed
    acc_mean    = delta_v / duration if duration and duration > 0 else np.nan

    # 位移
    dx = df.iloc[end_idx]["enu_x"] - df.iloc[start_idx]["enu_x"]
    dy = df.iloc[end_idx]["enu_y"] - df.iloc[start_idx]["enu_y"]
    displacement = np.sqrt(dx**2 + dy**2)

    # 航向角
    heading_start = np.degrees(df.iloc[start_idx]["cum_heading"])
    heading_end   = np.degrees(df.iloc[end_idx]["cum_heading"])
    turn_deg = normalize_angle(np.radians(heading_end - heading_start)) * 180 / np.pi
    heading_rate = turn_deg / duration if duration and duration > 0 else np.nan

    if turn_deg > np.degrees(turn_th):
        turn_label = "左转"
    elif turn_deg < -np.degrees(turn_th):
        turn_label = "右转"
    else:
        turn_label = "直行"

    return {
        "seq_start": start_seq,
        "seq_end": end_seq,
        "start_time": round(start_time, 2),
        "end_time": round(end_time, 2),
        "duration_s": round(duration, 2) if duration else None,

        # 状态
        "state": state,

        # 速度信息
        "start_speed": round(start_speed, 3),
        "end_speed": round(end_speed, 3),
        "avg_speed": round(avg_speed, 3),
        "max_speed": round(max_speed, 3),
        "min_speed": round(min_speed, 3),
        "delta_v": round(delta_v, 3),
        "mean_acc": round(acc_mean, 3) if acc_mean else None,

        # 位移信息
        "displacement": round(displacement, 3),
        "dx": round(dx, 3),
        "dy": round(dy, 3),

        # 航向角信息
        "turn_label": turn_label,
        "heading_start": round(heading_start, 2),
        "heading_end": round(heading_end, 2),
        "turn_angle_deg": round(turn_deg, 2),
        "heading_rate": round(heading_rate, 3) if heading_rate else None
    }

# ========== 文件分析 ==========
def analyze_file(file_path):
    df = read_txt(file_path)

    # 计算速度与角度
    df["speed"] = np.sqrt(df["enu_vx"]**2 + df["enu_vy"]**2 + df["enu_vz"]**2)
    df["speed_smooth"] = df["speed"].rolling(window=5, center=True, min_periods=1).mean()
    df["delta_v"] = df["speed_smooth"].diff().fillna(0)
    df["delta_heading"] = df["enu_rz"].diff().fillna(0).apply(normalize_angle)
    df["cum_heading"] = df["delta_heading"].cumsum()

    # 分段逻辑
    segments = []
    start_idx = 0
    prev_state = get_motion_state(df.iloc[0]["speed_smooth"], df.iloc[0]["delta_v"])
    start_heading = df.iloc[0]["cum_heading"]

    for i in range(1, len(df)):
        state = get_motion_state(df.iloc[i]["speed_smooth"], df.iloc[i]["delta_v"])
        if state != prev_state and (i - start_idx) >= min_segment_len:
            seg = summarize_segment(df, start_idx, i-1, prev_state, start_heading)
            if seg:
                segments.append(seg)
            start_idx = i
            prev_state = state
            start_heading = df.iloc[i]["cum_heading"]

    if len(df) - start_idx >= min_segment_len:
        seg = summarize_segment(df, start_idx, len(df)-1, prev_state, start_heading)
        if seg:
            segments.append(seg)

    return segments

# ========== 主程序 ==========
def main():
    for file_path in glob.glob(os.path.join(input_dir, "*.txt")):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        segments = analyze_file(file_path)

        if not segments:
            print(f"{file_name} 没有有效分段，跳过")
            continue

        # 输出 CSV
        df_out = pd.DataFrame(segments)
        csv_path = os.path.join(output_dir, f"{file_name}.csv")
        df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 输出 JSON
        json_path = os.path.join(output_dir, f"{file_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        print(f"{file_name} 分析完成 → {csv_path}, {json_path}")

if __name__ == "__main__":
    main()
