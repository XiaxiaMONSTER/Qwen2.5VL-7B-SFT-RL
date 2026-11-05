#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量分析轨迹 TXT 与对应的 JSON 红绿灯信息
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd

# ---------------- 参数 ----------------
WINDOW = 30  # 前后多少行作为分析窗口
SPEED_DROP_TH = 0.2  # m/s
SPEED_INCREASE_TH = 0.2
TURN_HEADING_TH = 30  # 度

# ---------------- 工具函数 ----------------
def parse_frame_index(fn: str) -> int:
    m = re.search(r'frame[_-]?(\d+)', fn)
    if not m:
        raise ValueError(f"无法解析 frame index: {fn}")
    return int(m.group(1))

def map_frame_to_seq(frame_index: int) -> int:
    # 你的映射公式
    return (frame_index + 1) * 3 - 1

# def read_txt(txt_path: str) -> pd.DataFrame:
#     df = pd.read_csv(txt_path)
#     df['seq'] = df['seq'].astype(int)
#     for col in ['time', 'enu_vx', 'enu_vy', 'enu_vz', 'enu_rz']:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#     return df

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

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 按列直接算速度
    df['speed'] = np.sqrt(df['enu_vx']**2 + df['enu_vy']**2)
    df['dt'] = df['time'].diff().fillna(0)
    df['dv'] = df['speed'].diff().fillna(0)
    df['accel'] = df.apply(lambda r: r['dv']/r['dt'] if r['dt']>1e-6 else 0, axis=1)
    df['heading_deg'] = np.degrees(df['enu_rz'])
    return df


def find_closest_seq(df: pd.DataFrame, target_seq: int) -> int:
    # 去掉 NA，并转成 numpy int 数组
    seqs = df['seq'].dropna().astype(int).to_numpy()
    idx = np.argmin(np.abs(seqs - target_seq))
    return int(seqs[idx])


def extract_window(df: pd.DataFrame, center_seq: int, window: int = WINDOW) -> pd.DataFrame:
    if center_seq not in df['seq'].values:
        center_seq = find_closest_seq(df, center_seq)
    idx = df.index[df['seq'] == center_seq][0]
    start = max(0, idx - window)
    end = min(len(df)-1, idx + window)
    return df.iloc[start:end+1].reset_index(drop=True), center_seq

# def compute_features(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df['speed'] = np.sqrt(df['enu_vx']**2 + df['enu_vy']**2)
#     df['dt'] = df['time'].diff().fillna(0)
#     df['dv'] = df['speed'].diff().fillna(0)
#     df['accel'] = df.apply(lambda r: r['dv']/r['dt'] if r['dt']>1e-6 else 0, axis=1)
#     df['heading_deg'] = np.degrees(df['enu_rz'])
#     return df

def summarize_trajectory(df: pd.DataFrame, center_seq: int) -> (str, dict):
    center_idx = df.index[df['seq']==center_seq][0]
    k = 5
    pre_speed = df['speed'].iloc[max(0,center_idx-k):center_idx].mean()
    post_speed = df['speed'].iloc[center_idx+1:center_idx+1+k].mean()
    speed_drop = pre_speed - post_speed
    speed_increase = post_speed - pre_speed

    # 位移变化
    if {'enu_x','enu_y'}.issubset(df.columns):
        dx = df['enu_x'].iloc[-1] - df['enu_x'].iloc[0]
        dy = df['enu_y'].iloc[-1] - df['enu_y'].iloc[0]
        displacement = np.sqrt(dx**2 + dy**2)
    else:
        displacement = None

    # 朝向变化
    heading_change = df['heading_deg'].iloc[-1] - df['heading_deg'].iloc[0]

    summary_text = (
        f"在窗口内({df['time'].iloc[0]:.1f}s~{df['time'].iloc[-1]:.1f}s)，"
        f"车速均值由 {pre_speed:.2f} m/s 到 {post_speed:.2f} m/s，"
        f"{'下降' if speed_drop>0 else '上升'} {abs(speed_drop):.2f} m/s，"
        f"航向角变化 {heading_change:.1f}°"
    )
    if displacement is not None:
        summary_text += f"，位移约 {displacement:.1f} 米"

    return summary_text, {
        'pre_speed': pre_speed,
        'post_speed': post_speed,
        'speed_drop': speed_drop,
        'speed_increase': speed_increase,
        'heading_change': heading_change,
        'displacement': displacement
    }

def summarize_lamp(lamps: list) -> (str, dict):
    relevant = [l for l in lamps if l.get('relevance')=='relevant']
    if not relevant:
        return "无相关灯", {}
    lamp = relevant[0]
    text = f"有相关灯，类型:{lamp.get('type','?')}，方向:{lamp.get('direction','?')}，指示:{'/'.join(lamp.get('indication',[]))}，灯态:{lamp.get('state','?')}"
    return text, lamp

def analyze_relationship(features: dict, lamp: dict) -> str:
    state = lamp.get('state','').lower()
    ltype = lamp.get('type','')
    indication = lamp.get('indication',[])
    hdg = features.get('heading_change',0)
    drop = features.get('speed_drop',0)
    inc = features.get('speed_increase',0)

    # 先总结规律区间然后将帧落入区间

    if state in ['unknown','off'] or ltype!='路口处机动车红绿灯':
        return "灯态off或未知或非机动车红绿灯，无法判断！"

    rules = []

    # 灯态相关判断
    if state == 'yellow' and drop > SPEED_DROP_TH:
        rules.append("黄灯前明显减速，可能因灯态减速")
    if state == 'red' and drop > SPEED_DROP_TH: # 添加低速停止
        rules.append("红灯前明显减速，可能因灯态减速")
    if state == 'green' and inc > SPEED_INCREASE_TH:
        rules.append("绿灯前有加速迹象")
    if state == 'green' and '直行' in indication \
       and abs(hdg) < TURN_HEADING_TH \
       and drop < SPEED_DROP_TH and inc < SPEED_INCREASE_TH:
        rules.append("直行绿灯，车辆正常直行通过，各项指标变化微小")
    # 速度稳定为0左右，那就是停车了，大概率前面是个红灯，在等红灯

    # 转弯相关判断
    if '左' in indication and hdg > TURN_HEADING_TH:
        rules.append("车辆左转且灯指示左转或掉头，可能相关")
    if '右' in indication and hdg < -TURN_HEADING_TH:
        rules.append("车辆右转且灯指示右转，可能相关")

    # 其他通用判断
    if drop > SPEED_DROP_TH and not ((state == 'yellow') or (state == 'red')):
        rules.append("有明显减速，但灯态与指示不匹配，可能为防御性驾驶或其他原因")

    # 输出结果
    if rules:
        return "；".join(rules)
    else:
        return "未发现明显与灯相关的轨迹变化" 

# ---------------- 主批处理 ----------------
def process_all(json_dir: str, txt_dir: str, out_csv: str, out_json: str):
    results = []
    for json_file in sorted(os.listdir(json_dir)):
        if not json_file.endswith('.json'):
            continue
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        lamps = data.get('traffic_light', [])
        jpg_scene = data.get('scene', '未知场景')
        if jpg_scene=='正在经过路口':
            print(f"[跳过] “正在经过路口”场景: {json_file}")
            continue  # 跳过正在经过路口的场景
        relevant_lamps = [l for l in lamps if l.get('relevance')=='relevant']
        if not relevant_lamps:
            continue

        # 找对应 TXT
        base = json_file.split('_frame_')[0] + '.txt'
        txt_path = os.path.join(txt_dir, base)
        if not os.path.exists(txt_path):
            print(f"未找到对应 TXT: {base}")
            continue

        # seq 匹配
        frame_idx = parse_frame_index(json_file)
        seq = map_frame_to_seq(frame_idx)
        df = read_txt(txt_path)
        df_win, seq = extract_window(df, seq)
        df_feat = compute_features(df_win)

        traj_text, traj_feat = summarize_trajectory(df_feat, seq)
        lamp_text, lamp_info = summarize_lamp(lamps)
        relation = analyze_relationship(traj_feat, lamp_info)

        results.append({
            'json_name': json_file,
            'trajectory_summary': traj_text,
            'lamp_summary': lamp_text,
            'relationship_summary': relation,
            **traj_feat,
            **lamp_info
        })

    # 保存
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8-sig')
    # with open(out_json, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(results)} 条结果到 {out_csv} 和 {out_json}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('json_dir', default='/root/autodl-tmp/batch03_final_deliver',help='存放红绿灯 JSON 的文件夹') # /root/autodl-tmp/batch03_final_deliver
    ap.add_argument('txt_dir',default='/root/autodl-tmp/dl0805/imgpost' ,help='存放轨迹 TXT 的文件夹') # /root/autodl-tmp/dl0805/imgpost
    ap.add_argument('--out_csv', default='analysis_results5.csv')
    ap.add_argument('--out_json', default='analysis_results5.json')
    args = ap.parse_args()
    process_all(args.json_dir, args.txt_dir, args.out_csv, args.out_json)
