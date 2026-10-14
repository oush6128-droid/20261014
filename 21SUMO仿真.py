import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import traci
import traceback
from collections import deque

# ================= 路径配置 =================
SUMO_HOME = r"D:\SUMO"
if os.path.exists(SUMO_HOME):
    sys.path.append(os.path.join(SUMO_HOME, "tools"))

BASE_DIR = r"D:\SUMO\1111111111TEST"
SUMO_CONFIG = os.path.join(BASE_DIR, "1111111111.sumocfg")
CSV_FILE = os.path.join(BASE_DIR, "932-400m.csv")
MODEL_DIR = r"D:\1\saved_models_full_features"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.txt")
IDM_MODEL_DIR = r"D:\1\IDM111"

# 检查文件是否存在
print(f"模型文件存在: {os.path.exists(MODEL_PATH)}")
print(f"Scaler文件存在: {os.path.exists(SCALER_PATH)}")
print(f"特征文件存在: {os.path.exists(FEATURES_PATH)}")

# ================= 全局参数 =================
STEP_LENGTH = 0.1
REPLAY_END_X = 400.0  # 前400m按轨迹重放
SIMULATION_END_TIME = 300.0
USE_GUI = True
TARGET_REAL_TIME_FACTOR = 1.0
DECISION_INTERVAL = 1.0
LANE_CHANGE_DURATION = 3.0
CLASS_NAMES = ['左换道', '右换道', '跟驰', '自由直行']
STYLE_MAP = {'激进': 1, '谨慎': 2, '保守': 3}
VEH_TYPE_MAP = {1: "type1", 2: "type2", 3: "type3", 4: "type4", 5: "type5"}
ROUTE_ID = "route0"
DESTINATION_X = 1136.6971  # 总长度约1136m
TIME_STEPS = 10  # LSTM需要10帧历史

# ================= 关键：恢复您原始的正确坐标转换参数 =================
# SUMO net.xml 车道Y坐标：
# lane0 (最右): y = -8.75
# lane1 (中间): y = -5.25
# lane2 (最左): y = -1.75
SUMO_LANE_Y = [-8.75, -5.25, -1.75]  # SUMO中每个车道的Y坐标
CSV_LANE_WIDTH = 3.5  # 假设CSV中车道宽度也是3.5米


def convert_csv_to_sumo_y(csv_y, csv_lane):
    """
    恢复您原始的正确转换函数
    CSV: 最右车道Y=0，向左递增（正方向）
    SUMO: 最右车道Y=-8.75，向左递增（负值变小）
    """
    # 方法1：根据CSV车道号直接映射到SUMO车道Y坐标（优先使用）
    if csv_lane == 0:  # CSV最左
        return SUMO_LANE_Y[2]  # SUMO最左
    elif csv_lane == 1:  # CSV中间
        return SUMO_LANE_Y[1]  # SUMO中间
    elif csv_lane == 2:  # CSV最右
        return SUMO_LANE_Y[0]  # SUMO最右
    else:
        # 方法2：计算相对位置（如果车道号不标准）
        csv_max_y = 7.0  # 2 lanes * 3.5m
        sumo_min_y = -8.75
        sumo_max_y = -1.75
        sumo_span = sumo_max_y - sumo_min_y

        # 归一化CSV Y到[0,1]
        normalized_y = min(max(csv_y / csv_max_y, 0.0), 1.0)
        # 映射到SUMO Y范围
        return sumo_min_y + normalized_y * sumo_span


# ================= 恢复您原始的车辆添加函数 =================
def safe_add_vehicle_now(sumo_vid, route_id, veh_type, depart_lane, depart_pos):
    """安全添加车辆 - 使用正确的车道索引（恢复您原始的正确版本）"""
    try:
        if sumo_vid in traci.vehicle.getIDList():
            return True

        # 严格校验车道索引 (0,1,2)
        if not isinstance(depart_lane, int) or depart_lane < 0 or depart_lane > 2:
            print(f"⚠️ 车辆 {sumo_vid} 的出发车道 {depart_lane} 无效，将使用默认值 'best'")
            lane_str = "best"
        else:
            lane_str = str(depart_lane)  # SUMO车道索引

        # 校验位置
        if not isinstance(depart_pos, (int, float)) or depart_pos < 0.0:
            print(f"⚠️ 车辆 {sumo_vid} 的出发位置 {depart_pos} 无效，将使用默认值 'base'")
            pos_str = "base"
        else:
            pos_str = str(max(1.0, depart_pos))

        # print(f"DEBUG 添加: {sumo_vid}, 目标SUMO车道={lane_str}, 位置={pos_str}")

        # 添加车辆
        traci.vehicle.add(
            vehID=sumo_vid,
            typeID=veh_type,
            routeID=route_id,
            depart=None,
            departLane=lane_str,
            departPos=pos_str,
            departSpeed="0"
        )

        # 等待车辆出现
        for _ in range(100):
            if sumo_vid in traci.vehicle.getIDList():
                traci.vehicle.setLaneChangeMode(sumo_vid, 0)  # 禁止自动换道
                traci.vehicle.setSpeedMode(sumo_vid, 0)  # 禁用速度控制
                return True
            time.sleep(0.01)

        return False
    except traci.TraCIException as e:
        print(f"⚠️ 添加车辆 {sumo_vid} 失败: {e}")
        return False


# ================= 恢复您原始的轨迹应用函数 =================
def apply_trajectory_frame(veh_id, frame_data):
    """强制移动车辆到轨迹点 - 恢复您原始的正确版本"""
    try:
        if veh_id not in traci.vehicle.getIDList():
            return

        x = frame_data['质心X']
        csv_y = frame_data['质心Y']
        csv_lane = int(frame_data['车道'])

        # 关键修复1：转换Y坐标到SUMO坐标系（使用您原始的正确函数）
        sumo_y = convert_csv_to_sumo_y(csv_y, csv_lane)

        # 关键修复2：moveToXY的lane参数设为-1，让SUMO自动根据坐标确定车道
        traci.vehicle.moveToXY(veh_id, "", -1, x, sumo_y, keepRoute=0)

        # 设置速度
        speed = frame_data['纵向速度']
        traci.vehicle.setSpeed(veh_id, speed)

    except Exception as e:
        print(f"❌ 应用轨迹失败 {veh_id}: {e}")
        traceback.print_exc()


# ================= LSTM 模型架构（从第二个代码复制） =================
class EnhancedMultiVehicleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=4, num_styles=3, style_embed_dim=32):
        super().__init__()
        self.style_embedding = nn.Embedding(num_styles, style_embed_dim)

        self.feature_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
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
        gamma = film_params[:, :128].unsqueeze(1)
        beta = film_params[:, 128:].unsqueeze(1)
        modulated = gamma * attn_out + beta

        final_out = modulated[:, -1, :]
        logits = self.classifier(final_out)

        return logits


# ================= LSTM 模型加载（从第二个代码复制，正确版本） =================
lstm_model = None
scaler = None
feature_names = []
input_size = 0
model_config = {}
lstm_loaded = False


def load_lstm_model():
    """正确加载EnhancedMultiVehicleLSTM模型"""
    global lstm_model, scaler, feature_names, input_size, model_config, lstm_loaded

    if not all([os.path.exists(MODEL_PATH), os.path.exists(SCALER_PATH), os.path.exists(FEATURES_PATH)]):
        print("❌ LSTM文件缺失")
        return False

    try:
        # 加载特征名称
        with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
            feature_names = [line.strip() for line in f.readlines() if line.strip()]
        input_size = len(feature_names)
        print(f"✅ 加载 {input_size} 个特征")
        # print("特征列表:", feature_names[:10], "...")

        # 加载标准化器
        scaler = joblib.load(SCALER_PATH)
        print("✅ Scaler加载成功")

        # 加载模型检查点
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        model_config = checkpoint['config']
        print(f"✅ 模型配置加载成功: {model_config}")

        # 创建模型
        lstm_model = EnhancedMultiVehicleLSTM(
            input_size=model_config['INPUT_SIZE'],
            hidden_size=model_config['HIDDEN_SIZE'],
            num_layers=model_config['NUM_LAYERS'],
            num_classes=model_config['NUM_CLASSES'],
            num_styles=model_config['NUM_STYLES'],
            style_embed_dim=model_config['STYLE_EMBED_DIM']
        )

        # 加载权重
        state_dict = checkpoint['model_state_dict']
        lstm_model.load_state_dict(state_dict)
        lstm_model.eval()

        lstm_loaded = True
        print(f"✅ LSTM模型加载成功！训练轮次: {checkpoint['epoch'] + 1}")
        return True

    except Exception as e:
        print(f"❌ LSTM加载失败: {e}")
        traceback.print_exc()
        return False


# ================= IDM 参数管理（从第二个代码复制） =================
class IDMManager:
    def __init__(self):
        self.idm_params = {}
        self.style_mapping = {
            "激进自信型": "Aggressive",
            "谨慎效率型": "Cautious",
            "保守跟随型": "Conservative"
        }
        self.mode_mapping = {
            "跟驰": "CarFollowing",
            "自由行驶": "FreeDriving"
        }
        self.load_all_idm_models()

    def load_all_idm_models(self):
        """加载所有6种IDM参数组合"""
        for style_cn, style_en in self.style_mapping.items():
            for mode_cn, mode_en in self.mode_mapping.items():
                try:
                    model_path = os.path.join(IDM_MODEL_DIR, f"IDM_{style_en}_{mode_en}.pth")
                    if not os.path.exists(model_path):
                        print(f"❌ IDM文件不存在: {model_path}")
                        continue

                    checkpoint = torch.load(model_path, map_location='cpu')
                    params = {k: v.item() for k, v in checkpoint['params'].items()}
                    self.idm_params[(style_cn, mode_cn)] = params
                    print(f"✅ 加载IDM参数: {style_cn}-{mode_cn}")
                except Exception as e:
                    print(f"❌ 加载IDM失败 {style_cn}-{mode_cn}: {e}")

    def get_idm_params(self, driving_style, mode):
        """获取IDM参数"""
        key = (driving_style, mode)
        if key in self.idm_params:
            return self.idm_params[key]
        else:
            print(f"⚠️ 未找到IDM参数: {key}，使用默认参数")
            return self.idm_params.get(("激进自信型", "跟驰"), {
                "a_max": 0.9127, "b": 2.3181, "v0": 22.6785, "T": 0.8942, "s0": 6.1433, "delta": 3.9013
            })


# ================= 车辆管理器（简化版，专注轨迹重放） =================
class VehicleManager:
    def __init__(self, idm_manager):
        self.vehicle_states = {}
        self.idm_manager = idm_manager
        self.feature_history = {}  # 每辆车10帧历史特征

    def get_sumo_veh_id(self, real_vid):
        return f"v_{real_vid}"

    def initialize_vehicle(self, real_vid, sumo_vid, first_frame):
        """初始化车辆状态"""
        style_str = first_frame.get('驾驶风格', '激进自信型')
        if style_str not in ["激进自信型", "谨慎效率型", "保守跟随型"]:
            style_str = '激进自信型'

        self.vehicle_states[sumo_vid] = {
            'real_id': real_vid,
            'traj_frames': [],  # 轨迹帧
            'traj_index': 0,
            'control_mode': 'replay',  # 'replay' or 'model'
            'last_decision_time': -1.0,
            'current_decision': 3,  # 3=自由直行
            'style': style_str,
            'last_position': (first_frame['质心X'], first_frame['质心Y'])
        }

        self.feature_history[sumo_vid] = deque(maxlen=TIME_STEPS)  # 10帧历史

    def extract_features_for_lstm(self, veh_id, current_frame):
        """提取LSTM所需特征"""
        if not lstm_loaded:
            return None

        try:
            features = {}
            for feature_name in feature_names:
                if feature_name in current_frame:
                    # 处理特殊字段
                    if 'ID' in feature_name and current_frame[feature_name] == -1:
                        features[feature_name] = -1
                    elif '间距' in feature_name or '距离' in feature_name:
                        features[feature_name] = current_frame[feature_name] if current_frame[
                                                                                    feature_name] > 0 else 1000.0
                    elif '速度' in feature_name or '加速度' in feature_name:
                        features[feature_name] = current_frame[feature_name] if current_frame[
                                                                                    feature_name] > -1 else 0.0
                    else:
                        features[feature_name] = current_frame[feature_name]
                else:
                    # 从SUMO获取或使用默认值
                    if feature_name == '质心X':
                        features[feature_name] = traci.vehicle.getPosition(veh_id)[0]
                    elif feature_name == '质心Y':
                        # 转换SUMO Y到CSV Y
                        sumo_y = traci.vehicle.getPosition(veh_id)[1]
                        csv_y = sumo_y + 8.75  # 简单转换
                        features[feature_name] = csv_y
                    elif feature_name == '车道':
                        # SUMO车道索引转CSV车道
                        sumo_lane = traci.vehicle.getLaneIndex(veh_id)
                        csv_lane = 2 - sumo_lane  # SUMO:0=右,2=左 -> CSV:0=左,2=右
                        features[feature_name] = csv_lane
                    elif feature_name == '纵向速度':
                        features[feature_name] = traci.vehicle.getSpeed(veh_id)
                    elif feature_name == '纵向加速度':
                        features[feature_name] = traci.vehicle.getAcceleration(veh_id)
                    elif feature_name == '车头时距':
                        features[feature_name] = self.calculate_headway(veh_id)
                    elif feature_name == '车型':
                        features[feature_name] = self.get_vehicle_type_code(veh_id)
                    elif feature_name == '目的地X':
                        features[feature_name] = DESTINATION_X
                    elif feature_name == '目的地车道':
                        features[feature_name] = 0  # CSV最左车道
                    else:
                        # 尝试从车辆状态获取
                        features[feature_name] = 0.0

            # 构建特征向量
            feature_vector = np.array([features[name] for name in feature_names], dtype=np.float32)

            # 缓存历史
            self.feature_history[veh_id].append(feature_vector.copy())

            return feature_vector

        except Exception as e:
            print(f"❌ 特征提取失败 {veh_id}: {e}")
            return None

    def calculate_headway(self, veh_id):
        """计算车头时距"""
        leader = traci.vehicle.getLeader(veh_id)
        if leader is None:
            return 999.0

        leader_id, distance = leader
        ego_speed = traci.vehicle.getSpeed(veh_id)
        if ego_speed < 0.1:
            return 999.0

        return distance / ego_speed

    def get_vehicle_type_code(self, veh_id):
        """获取车型代码"""
        vtype = traci.vehicle.getTypeID(veh_id)
        type_map = {"type1": 1, "type2": 2, "type3": 3, "type4": 4, "type5": 5}
        return type_map.get(vtype, 1)

    def make_decision(self, veh_id, current_time, current_frame):
        """使用LSTM做决策（需要10帧历史）"""
        if not lstm_loaded or len(self.feature_history[veh_id]) < TIME_STEPS:
            return self.rule_based_decision(veh_id), 0.0

        try:
            # 获取10帧历史特征
            history_features = np.array(list(self.feature_history[veh_id])[-TIME_STEPS:])

            # 标准化
            history_scaled = scaler.transform(history_features)

            # 转换为tensor
            feat_tensor = torch.tensor(history_scaled, dtype=torch.float32).unsqueeze(0)  # [1, 10, input_size]

            # 获取驾驶风格代码
            state = self.vehicle_states[veh_id]
            style_code = self.get_driving_style_code(state['style'])
            style_tensor = torch.tensor([style_code - 1], dtype=torch.long)  # 0,1,2

            # 预测
            with torch.no_grad():
                outputs = lstm_model(feat_tensor, style_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(outputs, dim=1).item()
                confidence = probs[0][pred_class].item()

            print(f"🚗 {veh_id} 决策: {CLASS_NAMES[pred_class]}, 置信度={confidence:.3f}")
            return pred_class, confidence

        except Exception as e:
            print(f"❌ LSTM决策失败 {veh_id}: {e}")
            traceback.print_exc()
            return self.rule_based_decision(veh_id), 0.0

    def get_driving_style_code(self, style_str):
        """获取驾驶风格代码（1,2,3）"""
        style_map = {
            "激进自信型": 1,
            "谨慎效率型": 2,
            "保守跟随型": 3
        }
        return style_map.get(style_str, 1)

    def rule_based_decision(self, veh_id):
        """基于规则的备用决策"""
        try:
            current_lane = traci.vehicle.getLaneIndex(veh_id)
            current_x = traci.vehicle.getPosition(veh_id)[0]
            speed = traci.vehicle.getSpeed(veh_id)

            # 接近目的地时保持车道
            if current_x > DESTINATION_X - 50:
                return 3  # 自由直行

            # 速度较慢且不在最左车道，尝试左换道
            if speed < 20 and current_lane < 2:
                return 0  # 左换道

            # 在最左车道，保持直行
            if current_lane == 2:
                return 3

            # 默认跟驰
            return 2

        except:
            return 3  # 默认自由直行

    def execute_decision(self, veh_id, decision_idx):
        """执行决策"""
        try:
            current_lane = traci.vehicle.getLaneIndex(veh_id)
            current_x = traci.vehicle.getPosition(veh_id)[0]

            if decision_idx == 0 and current_lane < 2:  # 左换道
                target_lane = current_lane + 1
                traci.vehicle.changeLane(veh_id, target_lane, LANE_CHANGE_DURATION)
                print(f"⚡ {veh_id} 瞬间左换道: {current_lane} -> {target_lane} (X={current_x:.1f})")
                return "跟驰"  # 换道后通常是跟驰

            elif decision_idx == 1 and current_lane > 0:  # 右换道
                target_lane = current_lane - 1
                traci.vehicle.changeLane(veh_id, target_lane, LANE_CHANGE_DURATION)
                print(f"⚡ {veh_id} 瞬间右换道: {current_lane} -> {target_lane} (X={current_x:.1f})")
                return "跟驰"  # 换道后通常是跟驰

            elif decision_idx == 2:  # 跟驰
                return "跟驰"

            elif decision_idx == 3:  # 自由直行
                return "自由行驶"

            return "跟驰"  # 默认

        except Exception as e:
            print(f"❌ 执行决策失败 {veh_id}: {e}")
            return "跟驰"

    def compute_idm_acceleration(self, veh_id, mode):
        """计算IDM加速度"""
        try:
            state = self.vehicle_states[veh_id]
            params = self.idm_manager.get_idm_params(state['style'], mode)

            v = traci.vehicle.getSpeed(veh_id)
            leader = traci.vehicle.getLeader(veh_id)

            if leader is None or mode == "自由行驶":
                # 自由行驶模式
                if 'a_max' in params and 'v0' in params and 'delta' in params:
                    a_max = params['a_max']
                    v0 = params['v0']
                    delta = params['delta']
                    return a_max * (1 - (v / v0) ** delta)
                else:
                    # 使用默认自由行驶参数
                    return 1.5 * (1 - (v / 20.0) ** 4)

            # 跟驰模式
            leader_id, gap = leader
            dv = v - traci.vehicle.getSpeed(leader_id)

            a_max = params['a_max']
            b = params['b']
            v0 = params['v0']
            T = params['T']
            s0 = params['s0']
            delta = params['delta']

            s_star = s0 + v * T + (v * dv) / (2 * np.sqrt(a_max * b))
            return a_max * (1 - (v / v0) ** delta - (s_star / max(gap, s0)) ** 2)

        except Exception as e:
            print(f"❌ IDM计算失败 {veh_id}: {e}")
            return 0.0


# ================= 主仿真（确保前400m轨迹重放，后750m模型控制） =================
def main():
    global lstm_loaded

    # 加载LSTM模型
    lstm_loaded = load_lstm_model()

    # 加载IDM参数
    idm_manager = IDMManager()

    # 加载轨迹
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV文件不存在: {CSV_FILE}")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
        print(f"✅ 读取CSV成功，共 {len(df)} 行")
    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        sys.exit(1)

    # 检查必要列
    required_cols = ['车辆ID', '时间戳', '质心X', '质心Y', '车道', '纵向速度']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必要列: {missing_cols}")
        print(f"可用列: {df.columns.tolist()}")
        sys.exit(1)

    # 按车辆ID和时间戳排序
    df = df.sort_values(['车辆ID', '时间戳'])

    # 构建轨迹字典
    traj_dict = {}
    first_app = {}
    for vid in df['车辆ID'].unique():
        vdf = df[df['车辆ID'] == vid].copy()
        traj_dict[vid] = vdf.to_dict('records')
        first_app[vid] = vdf.iloc[0]['时间戳']

    print(f"✅ 轨迹加载成功，包含 {len(traj_dict)} 辆车")

    # 初始化车辆管理器
    vehicle_manager = VehicleManager(idm_manager)

    # 启动SUMO
    sumo_bin = "sumo-gui" if USE_GUI else "sumo"
    print(f"🚀 启动SUMO: {sumo_bin}")

    try:
        traci.start([sumo_bin, "-c", SUMO_CONFIG, "--step-length", str(STEP_LENGTH), "--start"])
    except Exception as e:
        print(f"❌ SUMO启动失败: {e}")
        sys.exit(1)

    print("✅ SUMO启动成功")
    print("路线:", traci.route.getIDList())
    print("车型:", traci.vehicletype.getIDList())

    # 仿真状态
    sim_time = 0.0
    real_start = time.time()
    last_print = 0.0

    print("\n🚗 仿真开始（前400m轨迹重放，后750m模型控制）...")
    print("=" * 60)

    try:
        while sim_time < SIMULATION_END_TIME:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            # 添加到期车辆
            current_pending = sorted(first_app.items(), key=lambda x: x[1])
            for real_vid, appear_time in current_pending[:]:
                if appear_time <= sim_time + STEP_LENGTH / 2:
                    if real_vid not in traj_dict:
                        continue

                    frames = traj_dict[real_vid]
                    first_frame = frames[0]
                    sumo_vid = vehicle_manager.get_sumo_veh_id(real_vid)

                    # 车辆类型
                    veh_type_code = first_frame.get('车型', 1)
                    veh_type = VEH_TYPE_MAP.get(veh_type_code, "type1")

                    # 关键：恢复您原始的正确车道转换
                    csv_lane = int(first_frame['车道'])
                    sumo_lane = 2 - csv_lane  # CSV车道0(最左)->SUMO车道2(最左)
                    depart_x = first_frame['质心X']

                    # print(f"DEBUG Add: 车辆 {real_vid}, CSV车道={csv_lane}, 转换后SUMO车道={sumo_lane}, X={depart_x:.2f}")

                    if safe_add_vehicle_now(sumo_vid, ROUTE_ID, veh_type, sumo_lane, depart_x):
                        vehicle_manager.initialize_vehicle(real_vid, sumo_vid, first_frame)
                        vehicle_manager.vehicle_states[sumo_vid]['traj_frames'] = frames
                        # 应用第一帧
                        apply_trajectory_frame(sumo_vid, first_frame)
                        del first_app[real_vid]
                        print(f"🆕 新车 {real_vid} 添加成功，X={depart_x:.1f}, CSV车道={csv_lane}")

            # 更新所有车辆
            for sumo_vid in list(vehicle_manager.vehicle_states.keys()):
                if sumo_vid not in traci.vehicle.getIDList():
                    # print(f"⚠️ {sumo_vid} 不在仿真中，移除")
                    del vehicle_manager.vehicle_states[sumo_vid]
                    continue

                state = vehicle_manager.vehicle_states[sumo_vid]
                current_x, current_y = traci.vehicle.getPosition(sumo_vid)

                # 检查是否进入模型控制（关键：前400m轨迹重放，后750m模型控制）
                if state['control_mode'] == 'replay' and current_x >= REPLAY_END_X:
                    state['control_mode'] = 'model'
                    print(f"🎯 {sumo_vid} 进入模型控制 (X={current_x:.1f}, 车道={traci.vehicle.getLaneIndex(sumo_vid)})")

                # 轨迹重放模式（前400m）
                if state['control_mode'] == 'replay':
                    frames = state['traj_frames']
                    idx = state['traj_index']

                    # 找到当前时间对应的帧
                    while idx < len(frames) and frames[idx]['时间戳'] <= sim_time + 0.05:
                        idx += 1

                    if idx > 0:
                        current_frame = frames[idx - 1]
                        # 应用轨迹帧（使用您原始的正确函数）
                        apply_trajectory_frame(sumo_vid, current_frame)

                        # 提取特征（为后续模型控制准备）
                        vehicle_manager.extract_features_for_lstm(sumo_vid, current_frame)

                        state['traj_index'] = idx

                    # 检查是否轨迹结束
                    if idx >= len(frames):
                        state['control_mode'] = 'model'
                        print(f"🎬 {sumo_vid} 轨迹结束，进入模型控制 (X={current_x:.1f})")

                # 模型控制模式（后750m）
                elif state['control_mode'] == 'model':
                    # 获取当前帧信息
                    current_lane = traci.vehicle.getLaneIndex(sumo_vid)
                    current_speed = traci.vehicle.getSpeed(sumo_vid)

                    current_frame = {
                        '质心X': current_x,
                        '质心Y': current_y + 8.75,  # SUMO Y to CSV Y
                        '车道': 2 - current_lane,  # SUMO lane to CSV lane
                        '纵向速度': current_speed,
                        '纵向加速度': traci.vehicle.getAcceleration(sumo_vid)
                    }

                    # 提取特征
                    vehicle_manager.extract_features_for_lstm(sumo_vid, current_frame)

                    # 定期做决策
                    if sim_time - state['last_decision_time'] >= DECISION_INTERVAL:
                        decision_idx, confidence = vehicle_manager.make_decision(sumo_vid, sim_time, current_frame)
                        state['current_decision'] = decision_idx
                        state['last_decision_time'] = sim_time

                        # 执行决策并获取模式
                        mode = vehicle_manager.execute_decision(sumo_vid, decision_idx)
                        state['current_mode'] = mode

                    # 速度控制（使用IDM）
                    current_mode = state.get('current_mode', '跟驰')
                    acc = vehicle_manager.compute_idm_acceleration(sumo_vid, current_mode)
                    new_speed = max(0, current_speed + acc * STEP_LENGTH)
                    traci.vehicle.setSpeed(sumo_vid, new_speed)

            # 实时同步
            elapsed_real = time.time() - real_start
            target_elapsed = sim_time / TARGET_REAL_TIME_FACTOR
            if elapsed_real < target_elapsed:
                sleep_time = target_elapsed - elapsed_real
                if sleep_time > 0.001:  # 避免过短的sleep
                    time.sleep(sleep_time)

            # 状态输出
            if sim_time - last_print >= 10.0:
                active_vehicles = len(vehicle_manager.vehicle_states)
                print(f"\n⏱️ 仿真时间: {sim_time:.1f}s, 活跃车辆: {active_vehicles}")
                print("-" * 40)
                last_print = sim_time

    except KeyboardInterrupt:
        print("🛑 用户中断仿真")
    except Exception as e:
        print(f"💥 仿真异常: {e}")
        traceback.print_exc()
    finally:
        try:
            traci.close()
            print("\n🛑 仿真结束")
        except:
            pass


if __name__ == "__main__":
    main()