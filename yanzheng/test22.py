import os
import re
import json
import argparse
import pandas as pd

# ---------------- 工具函数 ----------------
def parse_frame_index(fn: str) -> int:
    m = re.search(r'frame[_-]?(\d+)', fn)
    if not m:
        raise ValueError(f"无法解析 frame index: {fn}")
    return int(m.group(1))

def map_frame_to_seq(frame_index: int) -> int:
    return (frame_index + 1) * 3 - 1

def summarize_lamp(lamps: list):
    relevant = [l for l in lamps if l.get('relevance')=='relevant']
    if not relevant:
        return "无相关灯", {}
    lamp = relevant[0]
    text = f"有相关灯，类型:{lamp.get('type','?')}，方向:{lamp.get('direction','?')}，" \
           f"指示:{'/'.join(lamp.get('indication',[]))}，灯态:{lamp.get('state','?')}"
    return text, lamp

def analyze_relationship(traj_feat: dict, lamp: dict) -> str:
    state = lamp.get('state','').lower()
    indication = lamp.get('indication',[])
    hdg = traj_feat.get('turn_angle_deg',0)
    drop = traj_feat.get('delta_v',0)
    ltype = lamp.get('type','')
    trturn=traj_feat.get('turn_label','')
    trstate=traj_feat.get('state','')

    if state in ['unknown','off'] or ltype!='路口处机动车红绿灯':
        return "灯态off或未知或非机动车红绿灯，无法判断！"

    rules = []
    state_rule=''
    turn_rule=''
    if state in ['yellow'] and trstate=='减速': # drop < 0
        state_rule=f"黄灯前减速，可能因灯态减速"
        #rules.append("黄灯前减速，可能因灯态减速")
    if state in ['red'] and trstate=='减速':
        state_rule=f"红灯前减速，可能因灯态减速"
        #rules.append("红灯前减速，可能因灯态减速")
    if state in ['red'] and trstate=='静止':
        state_rule=f"红灯前静止，可能因灯态静止"
        #rules.append("红灯前车辆静止，可能为红灯前停车")
    if state in ['green'] and (trstate=='加速' or trstate=='匀速'):
        state_rule=f"绿灯前加速或匀速行驶，可能因灯态绿灯"
        #rules.append("绿灯前匀速行驶或有加速迹象")
    if not state_rule:
        state_rule="未发现明显与灯态相关的轨迹变化"
    rules.append(state_rule)
    if '左' in indication and trturn=='左转': # hdg > 15
        turn_rule=f"车辆左转，灯指示左转，可能相关"
        #rules.append("车辆左转，灯指示左转，可能相关")
    if '右' in indication and trturn=='右转': # hdg < -15
        turn_rule=f"车辆右转，灯指示右转，可能相关"
    if '直行' in indication and trturn=='直行': # -15 < hdg < 15
        turn_rule=f"车辆直行，灯指示直行，可能相关"
        #rules.append("车辆直行，灯指示直行，可能相关")
    if not turn_rule:
        turn_rule="未发现明显与灯指示相关的转向变化"
    rules.append(turn_rule)
    if not rules:
        rules.append("完全不合规矩啊！")
    return "；".join(rules)

# ---------------- 主批处理 ----------------
def process_all(json_dir: str, seg_dir: str, out_csv: str):
    results = []
    for json_file in sorted(os.listdir(json_dir)):
        if not json_file.endswith('.json'):
            continue
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        jpg_scene = data.get('scene', '未知场景')
        if jpg_scene=='正在经过路口':
            print(f"[跳过] “正在经过路口”场景: {json_file}")
            continue  

        frame_idx = parse_frame_index(json_file)
        seq = map_frame_to_seq(frame_idx)

        # 找到对应的分段结果 csv
        base = json_file.split('_frame_')[0] + '.csv'
        seg_path = os.path.join(seg_dir, base)
        if not os.path.exists(seg_path):
            print(f"[警告] 找不到区间文件 {seg_path}")
            continue
        df_seg = pd.read_csv(seg_path)

        # 查找 seq 落在哪个区间
        hit = df_seg[(df_seg['seq_start'] <= seq) & (df_seg['seq_end'] >= seq)]
        if hit.empty:
            print(f"[警告] {json_file} 对应的 seq={seq} 未落入任何区间")
            continue
        traj_feat = hit.iloc[0].to_dict()

        lamps = data.get('traffic_light', [])

        for lamp in lamps:
            if lamp.get('relevance') != 'relevant':
                continue  # 跳过无关灯
            lamp_text = f"灯类型:{lamp.get('type','?')}，方向:{lamp.get('direction','?')}，" \
                        f"指示:{'/'.join(lamp.get('indication',[]))}，灯态:{lamp.get('state','?')}"
            
            relation = analyze_relationship(traj_feat, lamp)

            results.append({
                'json_name': json_file,
                'seq': seq,
                'traj_summary': f"{traj_feat['state']} {traj_feat['turn_label']} ({traj_feat['seq_start']}-{traj_feat['seq_end']})",
                'lamp_summary': lamp_text,
                'relationship_summary': relation,
                **traj_feat,
                **lamp
            })
        
        # lamp_text, lamp_info = summarize_lamp(lamps)
        # relation = analyze_relationship(traj_feat, lamp_info)

        # results.append({
        #     'json_name': json_file,
        #     'seq': seq,
        #     'traj_summary': f"{traj_feat['state']} {traj_feat['turn_label']} ({traj_feat['seq_start']}-{traj_feat['seq_end']})",
        #     'lamp_summary': lamp_text,
        #     'relationship_summary': relation,
        #     **traj_feat,
        #     **lamp_info
        # })

    pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"已保存 {len(results)} 条结果到 {out_csv}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--json_dir', default='/root/autodl-tmp/batch03_final_deliver',help='红绿灯 JSON 文件夹')
    ap.add_argument('--seg_dir',  default='/root/autodl-tmp/traffic_lights/yanzheng/anal_results1',help='区间规律 CSV 文件夹')
    ap.add_argument('--out_csv', default='analysis_with_segments4.csv')
    args = ap.parse_args()
    process_all(args.json_dir, args.seg_dir, args.out_csv)
