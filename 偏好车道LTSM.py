import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
from tqdm import tqdm
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ===================== 配置区域 =====================
BASE_PATH = r"D:\1TEST20260510\指标"
FILES = {
    "左换道": f"{BASE_PATH}\\左换道.csv",
    "右换道": f"{BASE_PATH}\\右换道.csv",
    "跟驰": f"{BASE_PATH}\\跟驰.csv",
    "自由直行": f"{BASE_PATH}\\自由直行.csv"
}
SAMPLE_ID_PREFIXES = {
    "左换道": "L",
    "右换道": "R",
    "自由直行": "ZY"
}

# 时序参数
TIME_STEPS = 10


FEATURES = [
    # ---------- 自车特征 (Ego Vehicle) ----------
    '质心X', '质心Y', '车道', '纵向速度', '纵向加速度',
    '横向速度', '横向加速度', '车头时距',

    # ---------- 前车特征 (Leading Vehicle) ----------
    '前车间距', '前车纵向速度', '前车纵向加速度', '前车车长', '前车车宽',

    # ---------- 左前车特征 (Left Leading) ----------
    '左前车间距', '左前车纵向速度', '左前车纵向加速度', '左前车车长', '左前车车宽',

    # ---------- 左后车特征 (Left Following) ----------
    '左后车间距', '左后车纵向速度', '左后车纵向加速度', '左后车车长', '左后车车宽',

    # ---------- 右前车特征 (Right Leading) ----------
    '右前车间距', '右前车纵向速度', '右前车纵向加速度', '右前车车长', '右前车车宽',

    # ---------- 右后车特征 (Right Following) ----------
    '右后车间距', '右后车纵向速度', '右后车纵向加速度', '右后车车长', '右后车车宽',

    # ---------- 目的地特征 ----------
    '目的地X'
]

# 自动计算输入维度
INPUT_SIZE = len(FEATURES)
print(f"特征总数: {INPUT_SIZE}")
print(f"特征列表: {FEATURES}")

# 模型超参数
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 4
NUM_STYLES = 3  # 新训练数据使用0、1、2
STYLE_EMBED_DIM = 32
NUM_PREFERRED_LANES = 4
PREFERRED_LANE_EMBED_DIM = 16
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['左换道', '右换道', '跟驰', '自由直行']

# 保存配置
SAVE_MODEL = True
SAVE_PATH = r"D:\2\saved_models_full_features"
SAVE_BEST_ONLY = True
SAVE_SCALER = True


def ensure_save_dir():
    if SAVE_MODEL:
        os.makedirs(SAVE_PATH, exist_ok=True)


def save_torch_file(obj, path):
    ensure_save_dir()
    with open(path, 'wb') as f:
        torch.save(obj, f)


def load_torch_file(path):
    with open(path, 'rb') as f:
        return torch.load(f)

# ===================== 数据加载与预处理 =====================
print("=" * 60)
print("正在加载数据...")
dfs = []
for i, (name, path) in enumerate(FILES.items()):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if name in SAMPLE_ID_PREFIXES:
        prefix = SAMPLE_ID_PREFIXES[name]
        df['样本ID'] = prefix + df['样本ID'].astype(str).str.replace(r'^CF', '', regex=True)
    df['decision_label'] = i
    dfs.append(df)
    print(f"已加载：{name}")

df = pd.concat(dfs, ignore_index=True)
print(f"总数据行数：{len(df)}")
print(f"总样本数：{df['样本ID'].nunique()}")

# 四车道偏好编号固定为0~3，可直接作为Embedding索引。
preferred_lane_values = pd.to_numeric(df['目的地车道'], errors='coerce').dropna()
if preferred_lane_values.empty:
    raise ValueError("目的地车道列没有有效数值")
preferred_lane_min = int(preferred_lane_values.min())
preferred_lane_max = int(preferred_lane_values.max())
if preferred_lane_min < 0 or preferred_lane_max >= NUM_PREFERRED_LANES:
    raise ValueError(
        f"目的地车道编号范围异常: {preferred_lane_min}~{preferred_lane_max}，"
        "四车道数据应使用0~3。"
    )
print(f"偏好车道编号范围: {preferred_lane_min}~{preferred_lane_max}")

style_values = pd.to_numeric(df['驾驶风格'], errors='coerce').dropna()
if style_values.empty:
    raise ValueError("驾驶风格列没有有效数值")
style_min = int(style_values.min())
style_max = int(style_values.max())
if style_min < 0 or style_max >= NUM_STYLES:
    raise ValueError(
        f"驾驶风格编号范围异常: {style_min}~{style_max}，当前数据应使用0~{NUM_STYLES - 1}。"
    )
print(f"驾驶风格编号范围: {style_min}~{style_max}")
print("=" * 60)

print("原始各类样本数：")
raw_label_counts = df.groupby('decision_label')['样本ID'].nunique().sort_index()
for label_id, count in raw_label_counts.items():
    print(f"  {label_id}-{CLASS_NAMES[int(label_id)]}: {count}")

sample_lengths = df.groupby(['decision_label', '样本ID']).size()
valid_length_counts = sample_lengths[sample_lengths == TIME_STEPS].groupby(level=0).size()
print(f"刚好满足 TIME_STEPS={TIME_STEPS} 的各类样本数：")
for label_id in range(NUM_CLASSES):
    count = int(valid_length_counts.get(label_id, 0))
    print(f"  {label_id}-{CLASS_NAMES[label_id]}: {count}")
print("=" * 60)

# 构建时序样本
print("构建含驾驶风格和偏好车道的时序样本...")


def build_sequences(df):
    seqs, styles, preferred_lanes, labels = [], [], [], []
    grouped_samples = df.groupby(['decision_label', '样本ID'], sort=False)
    for (decision_label, sid), tmp in tqdm(grouped_samples, total=grouped_samples.ngroups, desc="处理中"):
        tmp = tmp.sort_values('时间戳')
        if len(tmp) != TIME_STEPS:
            continue

        # 提取34维物理特征（偏好车道作为独立语义变量输入）
        try:
            feat_data = tmp[FEATURES].copy()

            # 填充缺失值：-1表示无车，填充为0或大值
            for col in FEATURES:
                if '间距' in col or '距离' in col:
                    # 间距类：无车填充为1000（表示很远）
                    feat_data[col] = feat_data[col].fillna(1000.0).replace(-1, 1000.0)
                elif '速度' in col or '加速度' in col:
                    # 速度/加速度类：无车填充为0
                    feat_data[col] = feat_data[col].fillna(0.0).replace(-1, 0.0)
                else:
                    # 其他：填充为0
                    feat_data[col] = feat_data[col].fillna(0.0).replace(-1, 0.0)

            style = int(round(tmp['驾驶风格'].iloc[0]))
            if not 0 <= style < NUM_STYLES:
                raise ValueError(f"驾驶风格越界: {style}")
            preferred_lane = tmp['目的地车道'].iloc[0]
            if pd.isna(preferred_lane):
                raise ValueError("目的地车道为空")
            preferred_lane = int(round(preferred_lane))
            if not 0 <= preferred_lane < NUM_PREFERRED_LANES:
                raise ValueError(f"偏好车道转换后越界: {preferred_lane}")

            seqs.append(feat_data.values.astype(np.float32))
            styles.append(style)
            preferred_lanes.append(preferred_lane)
            labels.append(int(decision_label))
        except Exception as e:
            print(f"样本 {sid} 处理失败: {e}")
            continue

    return (
        np.array(seqs),
        np.array(styles),
        np.array(preferred_lanes),
        np.array(labels)
    )


X, styles, preferred_lanes, y = build_sequences(df)
print(f"样本构建完成！形状：{X.shape}")
print("最终进入训练的各类样本数：")
final_label_ids, final_label_counts = np.unique(y, return_counts=True)
final_label_count_map = dict(zip(final_label_ids.astype(int), final_label_counts.astype(int)))
for label_id in range(NUM_CLASSES):
    print(f"  {label_id}-{CLASS_NAMES[label_id]}: {final_label_count_map.get(label_id, 0)}")

# 在数据送入CUDA前检查所有Embedding索引和分类标签，避免device-side assert。
def validate_index_range(name, values, upper_bound):
    if values.size == 0:
        raise ValueError(f"{name}为空")
    min_value = int(values.min())
    max_value = int(values.max())
    if min_value < 0 or max_value >= upper_bound:
        raise ValueError(
            f"{name}索引越界: {min_value}~{max_value}，合法范围为0~{upper_bound - 1}"
        )
    print(f"{name}索引范围: {min_value}~{max_value}")


validate_index_range("驾驶风格", styles, NUM_STYLES)
validate_index_range("偏好车道", preferred_lanes, NUM_PREFERRED_LANES)
validate_index_range("决策标签", y, NUM_CLASSES)

# 数据完整性检查
print("\n数据完整性检查：")
print("X 包含 NaN:", np.isnan(X).any())
print("X 包含 inf:", np.isinf(X).any())
if np.isnan(X).any() or np.isinf(X).any():
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
print("=" * 60)

# 数据标准化
print("数据标准化...")
scaler = StandardScaler()
X_reshaped = X.reshape(-1, INPUT_SIZE)
X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)

# 数据集划分
X_train, X_test, style_train, style_test, lane_train, lane_test, y_train, y_test = train_test_split(
    X_scaled, styles, preferred_lanes, y, test_size=0.3, random_state=42, stratify=y
)

# 转换为张量
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
style_train = torch.tensor(style_train, dtype=torch.long).to(DEVICE)
style_test = torch.tensor(style_test, dtype=torch.long).to(DEVICE)
lane_train = torch.tensor(lane_train, dtype=torch.long).to(DEVICE)
lane_test = torch.tensor(lane_test, dtype=torch.long).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, style_train, lane_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, style_test, lane_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


# ===================== 模型定义：增强版多车交互注意力LSTM =====================
class EnhancedMultiVehicleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # 风格嵌入
        self.style_embedding = nn.Embedding(NUM_STYLES, STYLE_EMBED_DIM)
        # 偏好车道嵌入
        self.preferred_lane_embedding = nn.Embedding(NUM_PREFERRED_LANES, PREFERRED_LANE_EMBED_DIM)

        # 特征投影层（将高维特征投影到合适维度）
        self.feature_proj = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        )

        # 双层LSTM
        self.lstm = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, bidirectional=False)
        self.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.dropout = nn.Dropout(0.3)

        # 多头自注意力
        self.multihead_attn = nn.MultiheadAttention(HIDDEN_SIZE, num_heads=4, batch_first=True)

        # 风格+偏好车道双条件融合与FiLM调制
        self.fusion_layer = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + STYLE_EMBED_DIM + PREFERRED_LANE_EMBED_DIM, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 2 * HIDDEN_SIZE),
            nn.Tanh()
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE // 2, NUM_CLASSES)
        )

    def forward(self, x, style, preferred_lane):
        batch_size = x.shape[0]

        # 1. 特征投影
        x_proj = self.feature_proj(x)

        # 2. LSTM编码
        lstm_out, _ = self.lstm(x_proj)
        lstm_out = self.norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        # 3. 自注意力
        attn_out, attn_weights = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm(attn_out)

        # 4. 风格与偏好车道嵌入融合
        style_emb = self.style_embedding(style)
        preferred_lane_emb = self.preferred_lane_embedding(preferred_lane)
        global_feat = torch.mean(attn_out, dim=1)
        fusion = torch.cat([global_feat, style_emb, preferred_lane_emb], dim=1)

        # 5. FiLM调制
        film_params = self.fusion_layer(fusion)
        gamma = film_params[:, :HIDDEN_SIZE].unsqueeze(1)
        beta = film_params[:, HIDDEN_SIZE:].unsqueeze(1)
        modulated = gamma * attn_out + beta

        # 6. 分类
        final_out = modulated[:, -1, :]
        logits = self.classifier(final_out)

        return logits, attn_weights


# ===================== 训练与评估 =====================
def train():
    if SAVE_MODEL:
        ensure_save_dir()
        if SAVE_SCALER:
            joblib.dump(scaler, f"{SAVE_PATH}/scaler.pkl")
            print(f"标准化器已保存至: {SAVE_PATH}/scaler.pkl")

        # 保存特征列表，方便后续预测使用
        with open(f"{SAVE_PATH}/features.txt", 'w', encoding='utf-8') as f:
            for feat in FEATURES:
                f.write(f"{feat}\n")
        print(f"特征列表已保存至: {SAVE_PATH}/features.txt")

    model = EnhancedMultiVehicleLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stop_counter = 0
    patience = 10

    print("\n开始训练增强版多车交互模型...")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [训练]")

        for x, style, preferred_lane, label in pbar:
            optimizer.zero_grad()
            outputs, _ = model(x, style, preferred_lane)
            loss = criterion(outputs, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        train_acc = correct / total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)

        # 验证阶段
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, style, preferred_lane, label in test_loader:
                outputs, _ = model(x, style, preferred_lane)
                loss = criterion(outputs, label)
                val_loss += loss.item()
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        val_acc = correct / total
        val_loss /= len(test_loader)
        scheduler.step(val_loss)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1:3d} | 训练ACC: {train_acc:.4f} | 验证ACC: {val_acc:.4f} | 验证Loss: {val_loss:.4f}")

        # 模型保存
        if SAVE_MODEL:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ensure_save_dir()
                save_torch_file({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'config': {
                        'INPUT_SIZE': INPUT_SIZE,
                        'HIDDEN_SIZE': HIDDEN_SIZE,
                        'NUM_LAYERS': NUM_LAYERS,
                        'NUM_CLASSES': NUM_CLASSES,
                        'NUM_STYLES': NUM_STYLES,
                        'STYLE_EMBED_DIM': STYLE_EMBED_DIM,
                        'NUM_PREFERRED_LANES': NUM_PREFERRED_LANES,
                        'PREFERRED_LANE_EMBED_DIM': PREFERRED_LANE_EMBED_DIM,
                        'TIME_STEPS': TIME_STEPS,
                        'FEATURES': FEATURES
                    }
                }, f"{SAVE_PATH}/best_model.pth")
                print(f"最佳模型已更新 (Epoch {epoch + 1})")
                early_stop_counter = 0
            else:
                early_stop_counter += 1

        # 早停
        if early_stop_counter >= patience:
            print(f"验证损失连续{patience}轮未下降，触发早停")
            break

    # ===================== 最终评估 =====================
    print("\n" + "=" * 60)
    print("开始最终评估...")
    if SAVE_MODEL and os.path.exists(f"{SAVE_PATH}/best_model.pth"):
        checkpoint = load_torch_file(f"{SAVE_PATH}/best_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("已加载最佳模型权重。")
    model.eval()

    all_pred, all_true, all_probs = [], [], []
    with torch.no_grad():
        for x, style, preferred_lane, label in test_loader:
            outputs, _ = model(x, style, preferred_lane)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 打印分类报告
    print("\n完整特征模型分类报告:")
    print(classification_report(all_true, all_pred, target_names=CLASS_NAMES, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('混淆矩阵 - 完整多车特征')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    if SAVE_MODEL:
        ensure_save_dir()
        plt.savefig(f"{SAVE_PATH}/confusion_matrix.png", dpi=300)
    plt.show()

    # 训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('准确率曲线')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('损失曲线')
    plt.legend()
    plt.tight_layout()
    if SAVE_MODEL:
        ensure_save_dir()
        plt.savefig(f"{SAVE_PATH}/training_curves.png", dpi=300)
    plt.show()

    return model, np.array(all_pred)


if __name__ == "__main__":
    model, predictions = train()
