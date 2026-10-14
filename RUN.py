import os
import sys
import numpy as np
import pandas as pd
import traci
import torch
import torch.nn as nn
from collections import defaultdict
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===================== 可调参数 =====================
SUMO_HOME = r"D:\\SUMO"
BASE_DIR = r"D:\\SUMO\\1111111111TEST"
SUMO_CONFIG = os.path.join(BASE_DIR, "1111111111.sumocfg")
CSV_FILE = os.path.join(BASE_DIR, "1845-400m.csv")
DECISION_LOG_FILE = os.path.join(BASE_DIR, "decision_log1845.csv")
MODEL_PATH = r"D:\\1\\saved_models_full_features\\best_model.pth"
SCALER_PATH = r"D:\\1\\saved_models_full_features\\scaler.pkl"

STEP_LENGTH = 0.1                 # 仿真步长（秒）
REPLAY_END_X = 400.0              # 复现区截止X坐标（米）
SIMULATION_END_TIME = 500        # 仿真总时长（秒）
USE_GUI = True                    # 是否打开SUMO图形界面
DECISION_INTERVAL = 1.0           # LSTM决策间隔（秒）
TIME_STEPS = 10                   # LSTM输入时序窗口长度（帧）
LC_DURATION = 3.0                 # 默认换道时长（秒），可按车辆覆盖
LC_COOLDOWN = 3.0                 # 换道冷却时间（秒）
LC_GAP_MIN = 15                  # 换道安全间距（米），目标车道前后均需大于此值
DEST_X = 1150                   # 目的地X坐标
VEH_DEFAULT_LENGTH = 4.5          # 默认车长（米）
VEH_DEFAULT_WIDTH = 1.8           # 默认车宽（米）

if os.path.exists(SUMO_HOME):
    sys.path.append(os.path.join(SUMO_HOME, "tools"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ===================== LSTM特征列表（顺序不能改） =====================
FEATURES = [
    '质心X', '质心Y', '车道', '纵向速度', '纵向加速度',
    '横向速度', '横向加速度', '车头时距',
    '前车间距', '前车纵向速度', '前车纵向加速度', '前车车长', '前车车宽',
    '左前车间距', '左前车纵向速度', '左前车纵向加速度', '左前车车长', '左前车车宽',
    '左后车间距', '左后车纵向速度', '左后车纵向加速度', '左后车车长', '左后车车宽',
    '右前车间距', '右前车纵向速度', '右前车纵向加速度', '右前车车长', '右前车车宽',
    '右后车间距', '右后车纵向速度', '右后车纵向加速度', '右后车车长', '右后车车宽',
    '目的地X', '目的地车道'
]
INPUT_SIZE = len(FEATURES)


# ===================== 辅助函数 =====================
def lane_from_y(y):
    """Y坐标 → 车道编号（0最右, 2最左）"""
    if y < -7.0:
        return 0
    elif y < -3.5:
        return 1
    else:
        return 2


def convert_csv_to_sumo_y(csv_y):
    csv_min_y, csv_max_y = 0.0, 10.5
    sumo_min_y, sumo_max_y = -10.5, 0.0
    csv_y = max(csv_min_y, min(csv_y, csv_max_y))
    norm = (csv_y - csv_min_y) / (csv_max_y - csv_min_y)
    return sumo_min_y + norm * (sumo_max_y - sumo_min_y)


def target_y_for_lane(lane):
    """车道编号 → SUMO Y坐标"""
    return convert_csv_to_sumo_y([1.75, 5.25, 8.75][lane])


def get_idm_params(style_name, mode):
    """获取IDM参数，带兜底"""
    default = IDM_PARAMS["谨慎效率型"]
    return IDM_PARAMS.get(style_name, default).get(mode, default["跟驰"])


# ===================== LSTM模型定义 =====================
class EnhancedMultiVehicleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=4,
                 num_styles=3, style_embed_dim=32):
        super().__init__()
        self.hidden_size = hidden_size

        self.style_embedding = nn.Embedding(num_styles, style_embed_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size + style_embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x, style):
        x_proj = self.feature_proj(x)
        lstm_out, _ = self.lstm(x_proj)
        lstm_out = self.norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        attn_out, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm(attn_out)

        style_emb = self.style_embedding(style)
        global_feat = torch.mean(attn_out, dim=1)

        fusion = torch.cat([global_feat, style_emb], dim=1)
        film_params = self.fusion_layer(fusion)
        gamma = film_params[:, :self.hidden_size].unsqueeze(1)
        beta = film_params[:, self.hidden_size:].unsqueeze(1)

        modulated = gamma * attn_out + beta
        final_out = modulated[:, -1, :]
        logits = self.classifier(final_out)
        return logits, None


# ===================== IDM参数 =====================
IDM_PARAMS = {
    "激进自信型": {
        "跟驰": {"a_max": 3.9127, "b": 2.3181, "v0": 23.6785, "T": 0.8942, "s0": 3.1433, "delta": 3.9013},
        "自由行驶": {"a_max": 3.5293, "v0": 20.4494, "delta": 4.1542}
    },
    "谨慎效率型": {
        "跟驰": {"a_max": 3.4832, "b": 2.1452, "v0": 21.0226, "T": 2.1072, "s0": 4.5751, "delta": 4.0503},
        "自由行驶": {"a_max": 3.0927, "v0": 18.3917, "delta": 4.0996}
    },
    "保守跟随型": {
        "跟驰": {"a_max": 3.6581, "b": 2.3331, "v0": 22.1780, "T": 1.1312, "s0": 3.6781, "delta": 3.9650},
        "自由行驶": {"a_max": 3.2670, "v0": 18.3511, "delta": 4.0634}
    }
}

# ===================== 车辆状态缓存 =====================
_prev_speeds = {}
_veh_static_cache = {}


def collect_vehicle_states():
    """批量收集所有车辆状态"""
    states = {}
    for vid in traci.vehicle.getIDList():
        try:
            x, y = traci.vehicle.getPosition(vid)
            states[vid] = {
                'x': x, 'y': y,
                'lane': traci.vehicle.getLaneIndex(vid),
                'speed': traci.vehicle.getSpeed(vid),
            }
        except:
            continue
        if vid not in _veh_static_cache:
            try:
                _veh_static_cache[vid] = {
                    'length': traci.vehicle.getLength(vid),
                    'width': traci.vehicle.getWidth(vid),
                }
            except:
                _veh_static_cache[vid] = {'length': VEH_DEFAULT_LENGTH, 'width': VEH_DEFAULT_WIDTH}
    return states


def get_surrounding_from_states(ego_id, ego_x, ego_lane, all_states, prev_speeds_dict=None):
    """从all_states查找周围车辆"""
    default = {'distance': 1000.0, 'speed': 0.0, 'acceleration': 0.0,
               'length': VEH_DEFAULT_LENGTH, 'width': VEH_DEFAULT_WIDTH}
    surrounding = {k: dict(default) for k in ['front', 'left_front', 'left_rear', 'right_front', 'right_rear']}

    spd_dict = prev_speeds_dict if prev_speeds_dict is not None else _prev_speeds
    for other, s in all_states.items():
        if other == ego_id:
            continue
        dist = s['x'] - ego_x
        if abs(dist) > 100:
            continue
        prev = spd_dict.get(other, s['speed'])
        acc = (s['speed'] - prev) / STEP_LENGTH
        static = _veh_static_cache.get(other, {'length': VEH_DEFAULT_LENGTH, 'width': VEH_DEFAULT_WIDTH})
        info = {'distance': abs(dist), 'speed': s['speed'], 'acceleration': acc,
                'length': static['length'], 'width': static['width']}

        if s['lane'] == ego_lane and dist > 0:
            if dist < surrounding['front']['distance']:
                surrounding['front'] = info
        elif s['lane'] == ego_lane + 1 and dist > 0:
            if dist < surrounding['left_front']['distance']:
                surrounding['left_front'] = info
        elif s['lane'] == ego_lane + 1 and dist < 0:
            if abs(dist) < surrounding['left_rear']['distance']:
                surrounding['left_rear'] = info
        elif s['lane'] == ego_lane - 1 and dist > 0:
            if dist < surrounding['right_front']['distance']:
                surrounding['right_front'] = info
        elif s['lane'] == ego_lane - 1 and dist < 0:
            if abs(dist) < surrounding['right_rear']['distance']:
                surrounding['right_rear'] = info
    return surrounding


def build_feature_dict(veh_id, veh_state, all_states, prev_speeds_dict=None, initial_lane=None):
    """构建35维特征字典"""
    x, y, speed, sumo_lane = veh_state['x'], veh_state['y'], veh_state['speed'], veh_state['lane']
    spd_dict = prev_speeds_dict if prev_speeds_dict is not None else _prev_speeds
    prev = spd_dict.get(veh_id, speed)
    acc = (speed - prev) / STEP_LENGTH

    surrounding = get_surrounding_from_states(veh_id, x, sumo_lane, all_states, prev_speeds_dict=spd_dict)
    headway = surrounding['front']['distance'] / speed if speed > 0.1 else 2.0

    if initial_lane == 0:
        dest_lane = 2
    elif initial_lane == 2:
        dest_lane = 0
    else:
        dest_lane = 1

    return {
        '质心X': x, '质心Y': y, '车道': sumo_lane,
        '纵向速度': speed, '纵向加速度': acc,
        '横向速度': 0.0, '横向加速度': 0.0, '车头时距': headway,
        '前车间距': surrounding['front']['distance'],
        '前车纵向速度': surrounding['front']['speed'],
        '前车纵向加速度': surrounding['front']['acceleration'],
        '前车车长': surrounding['front']['length'],
        '前车车宽': surrounding['front']['width'],
        '左前车间距': surrounding['left_front']['distance'],
        '左前车纵向速度': surrounding['left_front']['speed'],
        '左前车纵向加速度': surrounding['left_front']['acceleration'],
        '左前车车长': surrounding['left_front']['length'],
        '左前车车宽': surrounding['left_front']['width'],
        '左后车间距': surrounding['left_rear']['distance'],
        '左后车纵向速度': surrounding['left_rear']['speed'],
        '左后车纵向加速度': surrounding['left_rear']['acceleration'],
        '左后车车长': surrounding['left_rear']['length'],
        '左后车车宽': surrounding['left_rear']['width'],
        '右前车间距': surrounding['right_front']['distance'],
        '右前车纵向速度': surrounding['right_front']['speed'],
        '右前车纵向加速度': surrounding['right_front']['acceleration'],
        '右前车车长': surrounding['right_front']['length'],
        '右前车车宽': surrounding['right_front']['width'],
        '右后车间距': surrounding['right_rear']['distance'],
        '右后车纵向速度': surrounding['right_rear']['speed'],
        '右后车纵向加速度': surrounding['right_rear']['acceleration'],
        '右后车车长': surrounding['right_rear']['length'],
        '右后车车宽': surrounding['right_rear']['width'],
        '目的地X': DEST_X, '目的地车道': dest_lane
    }


def features_to_array(feat_dict, initial_lane=None):
    """特征字典 → numpy数组，NaN值用默认值填充"""
    defaults = {
        '间距': 1000.0, '速度': 0.0, '加速度': 0.0,
        '车长': VEH_DEFAULT_LENGTH, '车宽': VEH_DEFAULT_WIDTH,
        '目的地X': DEST_X
    }

    def get(key):
        val = feat_dict.get(key, np.nan)
        if not pd.isna(val):
            return val
        for suffix, dv in defaults.items():
            if suffix in key:
                return dv
        if key == '目的地车道':
            if initial_lane == 0:
                return 2
            elif initial_lane == 2:
                return 0
            return 1
        return 0.0

    return np.array([get(k) for k in FEATURES], dtype=np.float32)


# ===================== IDM速度控制 =====================
def idm_car_following(v, v_lead, spacing, params, dt=STEP_LENGTH):
    if spacing <= 0:
        return max(0.0, v - 7.0 * dt)
    s_star = params['s0'] + v * params['T'] + (v * (v - v_lead)) / (2 * np.sqrt(params['a_max'] * params['b']))
    a = params['a_max'] * (1 - (v / params['v0']) ** params['delta'] - (s_star / spacing) ** 2)
    a = max(-7.0, min(3.0, a))
    new_v = v + a * dt
    return max(0.0, min(new_v, 30.0))


def idm_free_driving(v, params, dt=STEP_LENGTH):
    a = params['a_max'] * (1 - (v / params['v0']) ** params['delta'])
    a = max(-7.0, min(3.0, a))
    new_v = v + a * dt
    return max(0.0, min(new_v, 30.0))


# ===================== 换道 =====================
def compute_lc_trajectory(origin_y, target_y, start_time, current_time, lc_duration):
    """余弦轨迹，返回 (new_y, is_complete)"""
    progress = (current_time - start_time) / lc_duration
    if progress >= 1.0:
        return target_y, True
    progress = max(0.0, progress)
    lateral_frac = (1 - np.cos(progress * np.pi)) / 2
    return origin_y + (target_y - origin_y) * lateral_frac, False


def execute_lane_change(veh_id, target_lane, current_time, current_lane, x, y,
                        all_states=None, label=None, lc_duration=None):
    """初始化换道，三层安全检查不通过返回None"""
    try:
        if current_lane == target_lane:
            return None
        target_lane = max(0, min(2, target_lane))

        # 统计各车道前后最近距离
        gaps = {}
        for lane in [0, 1, 2]:
            front_dist = 1000.0
            rear_dist = 1000.0
            for other, s in (all_states or {}).items():
                if other == veh_id or s['lane'] != lane:
                    continue
                d = s['x'] - x
                if d > 0 and d < front_dist:
                    front_dist = d
                elif d < 0 and abs(d) < rear_dist:
                    rear_dist = abs(d)
            gaps[lane] = {'front': front_dist, 'rear': rear_dist}

        cur_front = gaps[current_lane]['front']
        tgt_front = gaps[target_lane]['front']
        tgt_rear = gaps[target_lane]['rear']
        name = label if label else veh_id

        # 检查1：本车道前距 > 目标车道前距 → 取消换道
        if cur_front > tgt_front:
            print(f"⛔ {name} 换道取消 | 本车道前距({cur_front:.1f}m) > 目标车道前距({tgt_front:.1f}m)")
            return None

        # 检查2：中间车道(1)选择更优侧 → 若目标侧前距 < 另一侧前距，改目标
        if current_lane == 1:
            other_lane = 2 if target_lane == 0 else 0
            other_front = gaps[other_lane]['front']
            if tgt_front < other_front:
                print(f"🔄 {name} 目标调整 | 车道{target_lane}前距({tgt_front:.1f}m) < "
                      f"车道{other_lane}前距({other_front:.1f}m) → 改为车道{other_lane}")
                target_lane = other_lane
                tgt_front = other_front
                tgt_rear = gaps[other_lane]['rear']

        # 检查3：目标车道前后距离均 > LC_GAP_MIN 才允许
        if tgt_front <= LC_GAP_MIN or tgt_rear <= LC_GAP_MIN:
            print(f"⛔ {name} 换道取消 | 目标车道{target_lane} "
                  f"前距={tgt_front:.1f}m 后距={tgt_rear:.1f}m (需均>{LC_GAP_MIN}m)")
            return None

        dur = lc_duration if lc_duration is not None else LC_DURATION
        direction = "左" if target_lane > current_lane else "右"
        print(f"🚗 {name} {direction}换道 | {current_lane}→{target_lane} | {dur}s余弦轨迹")

        return {
            'start_time': current_time, 'origin_y': y,
            'target_y': target_y_for_lane(target_lane),
            'target_lane': target_lane, 'lc_duration': dur
        }
    except Exception as e:
        print(f"换道失败 {veh_id}: {e}")
        return None


# ===================== LSTM决策 =====================
def make_decision(model, history_buffer, style_idx, scaler, current_lane):
    """返回 0左/1右/2跟/3自"""
    if len(history_buffer) < TIME_STEPS:
        return 2

    arr = np.array(history_buffer[-TIME_STEPS:])
    if scaler:
        arr = scaler.transform(arr.reshape(-1, INPUT_SIZE)).reshape(arr.shape)

    x_tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    style_tensor = torch.tensor([style_idx], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(x_tensor, style_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        decision = np.argmax(probs)

    print(f"    [决策] 当前车道={current_lane}(0右,2左), "
          f"概率: 左={probs[0]:.3f}, 右={probs[1]:.3f}, 跟={probs[2]:.3f}, 自={probs[3]:.3f}")

    if current_lane == 0 and decision == 1:
        decision = 2
    if current_lane == 2 and decision == 0:
        decision = 2
    return decision


# ===================== 日志 =====================
def write_decision_log(record):
    try:
        df_log = pd.DataFrame([record])
        if not os.path.exists(DECISION_LOG_FILE):
            df_log.to_csv(DECISION_LOG_FILE, index=False, encoding='utf-8-sig')
        else:
            df_log.to_csv(DECISION_LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"⚠️ 写决策日志失败: {e}")


# ===================== 主仿真 =====================
def main():
    global _prev_speeds

    if not os.path.exists(CSV_FILE):
        print(f"CSV文件不存在: {CSV_FILE}")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
    required_cols = ['车辆ID', '时间戳', '质心X', '质心Y', '车道', '纵向速度', '驾驶风格']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"CSV缺少必要列: {missing}")
        sys.exit(1)

    df = df.sort_values(['车辆ID', '时间戳'])
    traj_dict = defaultdict(list)
    first_appearance = {}
    for _, row in df.iterrows():
        vid = row['车辆ID']
        traj_dict[vid].append(row.to_dict())
        if vid not in first_appearance:
            first_appearance[vid] = row['时间戳']

    sumo_bin = "sumo-gui" if USE_GUI else "sumo"
    traci.start([sumo_bin, "-c", SUMO_CONFIG, "--step-length", str(STEP_LENGTH),
                 "--lateral-resolution", "0.1", "--start"])
    print("SUMO启动成功")

    try:
        if "route0" not in traci.route.getIDList():
            traci.route.add("route0", ["edge1", "edge2"])
        if "DEFAULT_VEHTYPE" not in traci.vehicletype.getIDList():
            traci.vehicletype.copy("DEFAULT_VEHTYPE", "DEFAULT_VEHTYPE")
    except Exception as e:
        print(f"SUMO配置警告: {e}")

    # 加载模型
    model = None
    scaler = None
    try:
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            config = checkpoint.get('config', {})
            model = EnhancedMultiVehicleLSTM(
                input_size=config.get('INPUT_SIZE', INPUT_SIZE),
                hidden_size=config.get('HIDDEN_SIZE', 128),
                num_layers=config.get('NUM_LAYERS', 2),
                num_classes=config.get('NUM_CLASSES', 4),
                num_styles=config.get('NUM_STYLES', 3),
                style_embed_dim=config.get('STYLE_EMBED_DIM', 32),
            ).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"LSTM模型加载成功，输入维度={config.get('INPUT_SIZE', INPUT_SIZE)}")
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print("标准化器加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")

    style_map_text = {"激进自信型": 0, "谨慎效率型": 1, "保守跟随型": 2}
    style_idx_to_name = {0: "激进自信型", 1: "谨慎效率型", 2: "保守跟随型"}
    decision_names = ['左换道', '右换道', '跟驰', '自由直行']
    active_vehicles = {}
    sim_time = 0.0
    last_print = 0.0

    print(f"\n开始仿真：前{REPLAY_END_X}m轨迹复现，之后LSTM+IDM控制")
    print("=" * 60)

    try:
        while sim_time < SIMULATION_END_TIME:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            # --- 添加车辆 ---
            for real_vid, appear_time in list(first_appearance.items()):
                if appear_time <= sim_time + STEP_LENGTH / 2:
                    frames = traj_dict[real_vid]
                    first_frame = frames[0]
                    sumo_vid = f"v_{real_vid}"
                    try:
                        traci.vehicle.add(vehID=sumo_vid, typeID="DEFAULT_VEHTYPE", routeID="route0",
                                          depart=None, departLane="best",
                                          departPos=str(max(1.0, first_frame['质心X'])),
                                          departSpeed=str(first_frame.get('纵向速度', 0)))
                        traci.vehicle.setLaneChangeMode(sumo_vid, 0)
                        traci.vehicle.setSpeedMode(sumo_vid, 1)
                        init_x = first_frame['质心X']
                        init_y = convert_csv_to_sumo_y(first_frame['质心Y'])
                        traci.vehicle.moveToXY(sumo_vid, "", -1, init_x, init_y, keepRoute=0)

                        style_str = str(first_frame['驾驶风格']).strip()
                        if style_str.isdigit():
                            style_idx = int(style_str) - 1
                            style_name = style_idx_to_name.get(style_idx, "谨慎效率型")
                        else:
                            style_idx = style_map_text.get(style_str, 1)
                            style_name = style_str

                        active_vehicles[sumo_vid] = {
                            'real_id': real_vid, 'traj_frames': frames, 'current_idx': 0,
                            'history': [], 'last_decision_time': -1.0, 'current_decision': 2,
                            'style_idx': style_idx, 'style_name': style_name,
                            'lc_state': None,
                            'initial_lane': int(first_frame.get('车道', 1)),
                        }
                        print(f"添加车辆 {real_vid} 风格={style_name}")
                        del first_appearance[real_vid]
                    except Exception as e:
                        print(f"添加车辆 {real_vid} 失败: {e}")

            # --- 收集状态 ---
            all_states = collect_vehicle_states()
            for vid, s in all_states.items():
                traci_lane = s['lane']
                s['lane'] = lane_from_y(s['y'])
                if traci_lane != s['lane']:
                    print(f"⚠️ 车道修正: {vid} TraCI车道={traci_lane} Y={s['y']:.2f} → 修正为={s['lane']}")
            sumo_ids = set(all_states.keys())

            prev_speeds_snapshot = dict(_prev_speeds)

            # --- 更新车辆 ---
            for sumo_vid in list(active_vehicles.keys()):
                if sumo_vid not in sumo_ids:
                    del active_vehicles[sumo_vid]
                    continue

                state = active_vehicles[sumo_vid]
                vs = all_states[sumo_vid]
                x, y, sumo_lane, speed = vs['x'], vs['y'], vs['lane'], vs['speed']

                # 超出路网
                if x > DEST_X-10:
                    try:
                        traci.vehicle.remove(sumo_vid, reason=1)
                    except:
                        pass
                    del active_vehicles[sumo_vid]
                    print(f"[t={sim_time:.1f}s] 移除 {state['real_id']} (X={x:.1f}m, 超出路网)")
                    continue

                frames = state['traj_frames']
                cur_idx = state['current_idx']

                # ==== 复现区 ====
                if x < REPLAY_END_X and cur_idx < len(frames):
                    while cur_idx < len(frames) and frames[cur_idx]['时间戳'] <= sim_time + 0.01:
                        cur_idx += 1
                    if cur_idx > 0:
                        frame = frames[cur_idx - 1]
                        try:
                            traci.vehicle.moveToXY(sumo_vid, "", -1,
                                                   frame['质心X'],
                                                   convert_csv_to_sumo_y(frame['质心Y']),
                                                   keepRoute=0)
                            traci.vehicle.setSpeed(sumo_vid, frame['纵向速度'])
                        except Exception as e:
                            print(f"复现位置失败 {state['real_id']}: {e}")
                        else:
                            # 同步all_states，让控制区车辆可见
                            new_replay_y = convert_csv_to_sumo_y(frame['质心Y'])
                            all_states[sumo_vid]['x'] = frame['质心X']
                            all_states[sumo_vid]['y'] = new_replay_y
                            all_states[sumo_vid]['speed'] = frame['纵向速度']
                            all_states[sumo_vid]['lane'] = lane_from_y(new_replay_y)

                        feat = features_to_array(frame, initial_lane=state.get('initial_lane'))
                        state['history'].append(feat)
                        if len(state['history']) > TIME_STEPS * 2:
                            state['history'].pop(0)
                        state['current_idx'] = cur_idx
                    continue

                # ==== 控制区 ====
                traci.vehicle.setSpeedMode(sumo_vid, 0)

                feat_dict = build_feature_dict(sumo_vid, vs, all_states,
                                               prev_speeds_dict=prev_speeds_snapshot,
                                               initial_lane=state.get('initial_lane'))
                feat_arr = features_to_array(feat_dict, initial_lane=state.get('initial_lane'))
                state['history'].append(feat_arr)
                if len(state['history']) > TIME_STEPS * 2:
                    state['history'].pop(0)

                # LSTM决策
                if sim_time - state['last_decision_time'] >= DECISION_INTERVAL:
                    if model is not None and len(state['history']) >= TIME_STEPS:
                        y_expected_lane = lane_from_y(y)
                        if y_expected_lane != sumo_lane:
                            print(f"⚠️ 车道不一致! {state['real_id']} Y={y:.2f} sumo_lane={sumo_lane}")

                        decision = make_decision(model, state['history'], state['style_idx'],
                                                 scaler, sumo_lane)
                        state['current_decision'] = decision
                        state['last_decision_time'] = sim_time

                        # 写日志
                        history_frames = state['history'][-TIME_STEPS:]
                        for fi, frame_feat in enumerate(history_frames):
                            log_record = {
                                '时间戳': sim_time, '车辆ID': state['real_id'],
                                '驾驶风格': state['style_name'],
                                '决策结果': decision, '决策名称': decision_names[decision],
                                '帧序号': fi + 1,
                            }
                            for ki, k in enumerate(FEATURES):
                                log_record[k] = float(frame_feat[ki])
                            write_decision_log(log_record)

                        print(f"决策: {state['real_id']} X={x:.1f} {decision_names[decision]}")
                    else:
                        state['current_decision'] = 2

                # ==== 速度计算 ====
                lc_active = state.get('lc_state') is not None

                # 找本车道前车
                lead_v = None
                min_gap = float('inf')
                for other, s in all_states.items():
                    if other == sumo_vid:
                        continue
                    if s['lane'] == sumo_lane:
                        gap = s['x'] - x
                        if 0 <= gap < min_gap:
                            min_gap = gap
                            lead_v = other

                net_gap = float('inf')
                if lead_v is not None:
                    ego_len = _veh_static_cache.get(sumo_vid, {'length': VEH_DEFAULT_LENGTH})['length']
                    lead_len = _veh_static_cache.get(lead_v, {'length': VEH_DEFAULT_LENGTH})['length']
                    net_gap = min_gap - ego_len / 2 - lead_len / 2

                mode = "跟驰" if (lc_active or net_gap < 100 or state['current_decision'] == 2) else "自由行驶"

                if mode == "跟驰" and lead_v is not None:
                    lead_speed = all_states[lead_v]['speed']
                    eff_gap = net_gap  # 复用上面算好的净距
                    params = get_idm_params(state['style_name'], "跟驰")
                    new_speed = idm_car_following(speed, lead_speed, max(0.5, eff_gap), params)
                else:
                    params = get_idm_params(state['style_name'], mode if mode == "自由行驶" else "自由行驶")
                    new_speed = idm_free_driving(speed, params)

                new_x = x + new_speed * STEP_LENGTH

                # ==== 换道轨迹 ====
                if state.get('lc_state'):
                    lc = state['lc_state']
                    new_y, complete = compute_lc_trajectory(
                        lc['origin_y'], lc['target_y'], lc['start_time'], sim_time, lc['lc_duration'])
                    if complete:
                        print(f"  {state['real_id']} 换道完成 | 车道→{lc['target_lane']}")
                        state['lc_state'] = None
                        state['last_lc_complete_time'] = sim_time
                else:
                    new_y = -8.75 + sumo_lane * 3.5

                # ==== 新换道决策 ====
                if not state.get('lc_state') and state['current_decision'] in [0, 1]:
                    if sim_time - state.get('last_lc_complete_time', -LC_COOLDOWN) < LC_COOLDOWN:
                        state['current_decision'] = 2
                    else:
                        target_lane = sumo_lane + (1 if state['current_decision'] == 0 else -1)
                        target_lane = max(0, min(2, target_lane))
                        lc_state = execute_lane_change(sumo_vid, target_lane, sim_time,
                                                       sumo_lane, x, y,
                                                       all_states=all_states, label=state['real_id'])
                        if lc_state:
                            state['lc_state'] = lc_state
                        state['current_decision'] = 2

                # ==== 应用到SUMO ====
                traci.vehicle.moveToXY(sumo_vid, "", -1, new_x, new_y, keepRoute=0)
                traci.vehicle.setSpeed(sumo_vid, new_speed)
                all_states[sumo_vid]['x'] = new_x
                all_states[sumo_vid]['y'] = new_y
                all_states[sumo_vid]['speed'] = new_speed
                all_states[sumo_vid]['lane'] = lane_from_y(new_y)

            # 保存当前速度供下帧算加速度
            _prev_speeds = {vid: s['speed'] for vid, s in all_states.items()}

            # --- 状态输出 ---
            if sim_time - last_print >= 5.0:
                cnt_replay = sum(1 for v in active_vehicles
                                 if traci.vehicle.getPosition(v)[0] < REPLAY_END_X)
                cnt_ctrl = len(active_vehicles) - cnt_replay
                lane_counts = {0: 0, 1: 0, 2: 0}
                for v in active_vehicles:
                    try:
                        lane_counts[traci.vehicle.getLaneIndex(v)] += 1
                    except:
                        pass
                print(f"\n时间 {sim_time:.1f}s | 复现区:{cnt_replay} 控制区:{cnt_ctrl}")
                print(f"车道分布 (0最右,2最左): {lane_counts}")
                last_print = sim_time

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"仿真异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        traci.close()
        print("仿真结束")


if __name__ == "__main__":
    main()