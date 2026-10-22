
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset

BASE_PATH = Path(r"D:\1TEST20260510\指标")
SAVE_PATH = Path(r"D:\2\saved_models_R2_three_lane")
CLASS_NAMES = ["左换道", "右换道", "跟驰", "自由直行"]
FEATURES = [
    "质心X", "质心Y", "车道", "纵向速度", "纵向加速度", "横向速度", "横向加速度", "车头时距",
    "前车间距", "前车纵向速度", "前车纵向加速度", "前车车长", "前车车宽",
    "左前车间距", "左前车纵向速度", "左前车纵向加速度", "左前车车长", "左前车车宽",
    "左后车间距", "左后车纵向速度", "左后车纵向加速度", "左后车车长", "左后车车宽",
    "右前车间距", "右前车纵向速度", "右前车纵向加速度", "右前车车长", "右前车车宽",
    "右后车间距", "右后车纵向速度", "右后车纵向加速度", "右后车车长", "右后车车宽", "目的地X",
]
NEIGHBORS = ["前车", "左前车", "左后车", "右前车", "右后车"]
VARIANTS = ["full", "style_only", "preference_only", "unconditional", "direct_concat",
            "static_film", "no_cross_attention", "no_self_attention", "no_attention",
            "shuffled_preference", "transformer"]


@dataclass
class ModelConfig:
    input_size: int = 34
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_classes: int = 4
    num_styles: int = 3
    num_preferred_lanes: int = 3
    style_embed_dim: int = 32
    lane_embed_dim: int = 16
    time_steps: int = 10
    dropout: float = 0.3
    variant: str = "full"


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def integer_column(df, column, lower, upper):
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(values) | (values != np.floor(values)) | (values < lower) | (values >= upper)
    if bad.any():
        examples = df.loc[bad, ["车辆ID", column]].head(6).to_dict("records")
        raise ValueError(f"{column}必须是{lower}~{upper-1}的整数；异常示例：{examples}。不自动裁剪或合并车道。")
    df[column] = values.astype(np.int64)


def load_frames(data_dir, recording_col=None):
    frames, provenance = [], []
    required = FEATURES + ["样本ID", "车辆ID", "时间戳", "驾驶风格", "目的地车道"]
    for label, name in enumerate(CLASS_NAMES):
        path = Path(data_dir) / f"{name}.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"样本ID": str, "车辆ID": str})
        frame.columns = frame.columns.str.strip()
        missing = set(required + ([recording_col] if recording_col else [])) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name}缺少字段：{sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{path.name}没有样本")
        if frame[["样本ID", "车辆ID"]].isna().any().any():
            raise ValueError(f"{path.name}含空样本ID或车辆ID")
        if "决策标签" in frame and not frame["决策标签"].eq(name).all():
            raise ValueError(f"{path.name}中的决策标签与文件类别不一致")
        frame["_label"] = label
        # 类别前缀仅用于窗口ID，绝不加到物理车辆ID上。
        if recording_col:
            if frame[recording_col].isna().any():
                raise ValueError("录像来源字段存在缺失")
            recording = frame[recording_col].astype(str)
        else:
            recording = pd.Series("single_recording", index=frame.index)
        frame["_recording"] = recording
        # JSON编码防止分隔符碰撞；多录像复用车号时必须提供recording_col。
        frame["_vehicle"] = [json.dumps([r, v], ensure_ascii=False) for r, v in zip(recording, frame["车辆ID"])]
        frame["_sample"] = [json.dumps([r, label, s], ensure_ascii=False)
                             for r, s in zip(recording, frame["样本ID"])]
        frames.append(frame)
        provenance.append({"path": str(path.resolve()), "rows": len(frame),
                           "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.concat(frames, ignore_index=True), provenance


def audit_frames(df, terminal_x=None):
    """检查类别索引、车辆条件一致性、窗口长度和固定目标坐标。"""
    issues = []
    for col, upper in [("车道", 3), ("目的地车道", 3), ("驾驶风格", 3)]:
        try:
            integer_column(df, col, 0, upper)
        except ValueError as exc:
            issues.append(str(exc))
    for col in ["目的地车道", "驾驶风格"]:
        conflicts = df.groupby("_vehicle")[col].nunique(dropna=False)
        if conflicts.gt(1).any():
            issues.append(f"{int(conflicts.gt(1).sum())}个物理车辆的{col}在窗口间变化；固定条件不成立。"
                          "请核查数据增强/车号复用，必要时提供真实录像来源列，不能把样本ID当成车辆ID。")
    for col in ["_vehicle", "目的地车道", "驾驶风格"]:
        if df.groupby("_sample")[col].nunique(dropna=False).gt(1).any():
            issues.append(f"一个窗口内包含多个{col}")
    lengths = df.groupby("_sample").size()
    if lengths.ne(10).any():
        issues.append(f"{int(lengths.ne(10).sum())}个窗口不是10帧；本脚本不静默丢弃或跨窗口拼接")
    dest = pd.to_numeric(df["目的地X"], errors="coerce")
    if terminal_x is None:
        for recording, values in dest.groupby(df["_recording"]):
            if not np.isfinite(values).all() or values.max() - values.min() > 1e-3:
                issues.append(f"{recording}的目的地X不是固定断面坐标。请先核实上游生成逻辑，"
                              "或用--terminal-x指定经确认的固定断面；不能使用逐车换道终点坐标。")
    elif not np.isfinite(terminal_x):
        issues.append("terminal_x必须是有限数值")
    return {"rows": len(df), "vehicles": int(df["_vehicle"].nunique()),
            "windows": int(df["_sample"].nunique()),
            "lane_values": sorted(df["车道"].dropna().astype(str).unique().tolist()),
            "preference_values": sorted(df["目的地车道"].dropna().astype(str).unique().tolist()),
            "window_lengths": {str(k): int(v) for k, v in lengths.value_counts().items()},
            "recording_column_required_if_vehicle_ids_are_reused": True,
            "terminal_x_override": terminal_x, "issues": issues}


def preprocess_features(df, terminal_x=None, missing_distance=1000.0):
    """无车只按该邻车的mask处理，保留合法的负加速度/横向速度，包括-1。"""
    result = df.copy()
    for col in FEATURES + ["时间戳"]:
        result[col] = pd.to_numeric(result[col], errors="raise")
    if terminal_x is not None:
        result["目的地X"] = float(terminal_x)
    for neighbor in NEIGHBORS:
        id_col = neighbor + "ID"
        if id_col in result:
            ids = pd.to_numeric(result[id_col], errors="raise")
            absent = ids.isna() | ids.eq(-1)
        else:
            # 无ID时，用净间距缺失/-1识别；不以加速度符号判断车辆存在。
            gap = result[neighbor + "间距"]
            absent = gap.isna() | gap.eq(-1)
        cols = [neighbor + suffix for suffix in ["间距", "纵向速度", "纵向加速度", "车长", "车宽"]]
        present = result.loc[~absent, cols]
        if not np.isfinite(present.to_numpy(float)).all():
            raise ValueError(f"{neighbor}存在时仍有缺失/非有限特征，应修复上游数据")
        if (present[neighbor + "间距"] < 0).any() or (present[[neighbor+"车长", neighbor+"车宽"]] <= 0).any().any():
            raise ValueError(f"{neighbor}存在时净间距或车身尺寸异常")
        result.loc[absent, neighbor + "间距"] = missing_distance
        result.loc[absent, neighbor + "纵向速度"] = result.loc[absent, "纵向速度"]
        result.loc[absent, neighbor + "纵向加速度"] = result.loc[absent, "纵向加速度"]
        result.loc[absent, [neighbor + "车长", neighbor + "车宽"]] = 0.0
        if neighbor == "前车":
            result.loc[absent, "车头时距"] = missing_distance
    if not np.isfinite(result[FEATURES + ["时间戳"]].to_numpy(float)).all():
        raise ValueError("自车/有前车时距/时间戳包含缺失或无穷值；不自动以0掩盖")
    if (result["纵向速度"] < 0).any() or (result["车头时距"] < 0).any():
        raise ValueError("纵向速度或车头时距为负，请核实缺失编码")
    return result


def vehicle_split(df, seed=42):
    # 用车辆的主导动作类别分层，按物理车辆70/15/15划分，而非随机分窗口。
    # 通过--split-manifest可复用已有的车辆划分。
    vehicle_labels = pd.crosstab(df["_vehicle"], df["_label"]).idxmax(axis=1)
    vehicles = vehicle_labels.index.to_numpy()
    n_holdout = int(round(len(vehicles) * 0.15))
    if n_holdout < 4:
        raise ValueError("车辆太少，无法进行分层70/15/15划分")
    train_val, test = train_test_split(vehicles, test_size=n_holdout, random_state=seed,
                                     stratify=vehicle_labels.loc[vehicles])
    train, val = train_test_split(train_val, test_size=n_holdout, random_state=seed,
                                 stratify=vehicle_labels.loc[train_val])
    return {"train": sorted(train.tolist()), "validation": sorted(val.tolist()), "test": sorted(test.tolist())}


def validate_split(manifest, df):
    if set(manifest) != {"train", "validation", "test"}:
        raise ValueError("划分清单必须且只能包含train/validation/test三个车辆ID列表")
    groups = [set(manifest[k]) for k in ["train", "validation", "test"]]
    if any(len(g) != len(manifest[k]) for g, k in zip(groups, ["train", "validation", "test"])):
        raise ValueError("划分清单含重复车辆")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("物理车辆跨训练/验证/测试集，存在泄漏")
    if set.union(*groups) != set(df["_vehicle"]):
        raise ValueError("划分清单与本次车辆集合不一致")


def make_sequences(df, vehicle_ids):
    part = df.loc[df["_vehicle"].isin(vehicle_ids)].sort_values(["_sample", "时间戳"]).copy()
    sizes = part.groupby("_sample", sort=False).size()
    if sizes.empty or sizes.ne(10).any():
        raise ValueError("空划分或窗口长度不等于10")
    times = part["时间戳"].to_numpy().reshape(-1, 10)
    if not np.allclose(np.diff(times, axis=1), 0.1, atol=1e-4, rtol=0):
        raise ValueError("窗口不是连续0.1秒采样；请修复断点，不跨断点拼接")
    meta = part.iloc[9::10][["_sample", "_vehicle", "_label", "驾驶风格", "目的地车道", "时间戳", "质心X", "目的地X"]].reset_index(drop=True)
    # 当前代码只读取预构造窗口：动作起点、1秒预测标签和预处理因果性由上游负责。
    X = part[FEATURES].to_numpy(dtype=np.float32).reshape(-1, 10, 34)
    y = meta["_label"].to_numpy(np.int64)
    if set(np.unique(y)) != set(range(4)):
        raise ValueError("某个划分缺少动作类别，不能据此报告四类评价")
    return {"X": X, "style": meta["驾驶风格"].to_numpy(np.int64),
            "lane": meta["目的地车道"].to_numpy(np.int64), "y": y, "meta": meta}


class EnhancedMultiVehicleLSTM(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.variant not in VARIANTS:
            raise ValueError(c.variant)
        self.style_embedding = None if c.variant in ["unconditional", "preference_only"] else nn.Embedding(c.num_styles, c.style_embed_dim)
        self.preferred_lane_embedding = None if c.variant in ["unconditional", "style_only"] else nn.Embedding(c.num_preferred_lanes, c.lane_embed_dim)
        cond_dim = (c.style_embed_dim if self.style_embedding is not None else 0) + (c.lane_embed_dim if self.preferred_lane_embedding is not None else 0)
        self.feature_proj = nn.Sequential(nn.Linear(c.input_size, c.hidden_size), nn.ReLU(), nn.Linear(c.hidden_size, c.hidden_size))
        self.dropout = nn.Dropout(c.dropout)
        if c.variant == "transformer":
            self.position_embedding = nn.Parameter(torch.zeros(1, c.time_steps, c.hidden_size))
            nn.init.normal_(self.position_embedding, std=0.02)
            layer = nn.TransformerEncoderLayer(c.hidden_size, c.num_heads, 512, c.dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, c.num_layers)
            self.norm = nn.Identity()  
            self.multihead_attn = None
        else:
            self.lstm = nn.LSTM(c.hidden_size, c.hidden_size, c.num_layers, batch_first=True)
            self.norm = nn.LayerNorm(c.hidden_size)
            self.multihead_attn = None if c.variant in ["no_attention", "no_self_attention"] else nn.MultiheadAttention(c.hidden_size, c.num_heads, batch_first=True)
        use_cross = cond_dim > 0 and c.variant not in ["direct_concat", "static_film", "no_cross_attention", "no_attention"]
        self.condition_query = nn.Linear(cond_dim, c.hidden_size) if use_cross else None
        self.cross_attention = nn.MultiheadAttention(c.hidden_size, c.num_heads, batch_first=True) if use_cross else None
        self.fusion_layer = None
        if cond_dim and c.variant != "direct_concat":
            film_in = cond_dim if c.variant == "static_film" else c.hidden_size + cond_dim
            self.fusion_layer = nn.Sequential(nn.Linear(film_in, c.hidden_size), nn.ReLU(),
                                              nn.Linear(c.hidden_size, 2*c.hidden_size), nn.Tanh())
        classifier_in = c.hidden_size + (cond_dim if c.variant == "direct_concat" else 0)
        self.classifier = nn.Sequential(nn.Linear(classifier_in, c.hidden_size), nn.ReLU(), nn.Dropout(c.dropout),
                                        nn.Linear(c.hidden_size, 64), nn.ReLU(), nn.Linear(64, c.num_classes))

    def forward(self, x, style, preferred_lane, return_attention=False):
        c = self.config
        projected = self.feature_proj(x)
        if c.variant == "transformer":
            h = self.encoder(projected + self.position_embedding[:, :x.size(1)])
        else:
            h, _ = self.lstm(projected)
            h = self.dropout(self.norm(h))
        weights = None
        if self.multihead_attn is not None:
            h, weights = self.multihead_attn(h, h, h, need_weights=return_attention)
        conditions = []
        if self.style_embedding is not None:
            conditions.append(self.style_embedding(style))
        if self.preferred_lane_embedding is not None:
            conditions.append(self.preferred_lane_embedding(preferred_lane))
        if conditions:
            condition = torch.cat(conditions, dim=-1)
            if c.variant == "direct_concat":
                final = torch.cat([h[:, -1], condition], dim=-1)
            else:
                if c.variant == "static_film":
                    film_input = condition
                else:
                    if self.cross_attention is not None:
                        query = self.condition_query(condition).unsqueeze(1)
                        context, _ = self.cross_attention(query, h, h, need_weights=False)
                        context = context.squeeze(1)
                    elif c.variant == "no_attention":
                        context = h[:, -1]
                    else:
                        context = h.mean(dim=1)
                    film_input = torch.cat([context, condition], dim=-1)
                gamma, beta = self.fusion_layer(film_input).chunk(2, dim=-1)
                final = (gamma.unsqueeze(1) * h + beta.unsqueeze(1))[:, -1]
        else:
            final = h[:, -1]
        # 交叉熵接收原始logits；仅在评价/推理时softmax。
        return self.classifier(final), weights


class EpochUndersampler(Sampler):
    """每轮不放回抽取等量类别，轮次间重新抽样；验证和测试完全保留。"""
    def __init__(self, labels, seed):
        self.pools = [np.flatnonzero(labels == k) for k in range(4)]
        self.count = min(map(len, self.pools))
        self.seed, self.epoch = seed, 0
        if self.count == 0:
            raise ValueError("训练集缺少类别")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        indices = np.concatenate([rng.choice(pool, self.count, replace=False) for pool in self.pools])
        rng.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return 4*self.count


def make_loader(data, batch_size, device, training=False, balancing="undersample", seed=0):
    tensors = [torch.from_numpy(data[k]) for k in ["X", "style", "lane", "y"]]
    dataset = TensorDataset(*tensors)  # 数据留在CPU，按批次传到GPU。
    sampler = EpochUndersampler(data["y"], seed) if training and balancing == "undersample" else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=training and sampler is None, num_workers=0,
                      pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, style, lane, y in loader:
        out, _ = model(x.to(device, non_blocking=True), style.to(device, non_blocking=True), lane.to(device, non_blocking=True))
        logits.append(out.cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def expected_calibration_error(probabilities, labels, bins=15):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    # 恰为1的置信度也进入最后一个bin。
    bin_ids = np.minimum((confidence * bins).astype(int), bins-1)
    return float(sum(np.mean(bin_ids == b)*abs(correct[bin_ids == b].mean()-confidence[bin_ids == b].mean())
                     for b in range(bins) if np.any(bin_ids == b)))


def metrics(logits, labels):
    labels = np.asarray(labels)
    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    cm = confusion_matrix(labels, predictions, labels=range(4))
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=range(4), average="macro", zero_division=0)),
            "nll": float(nn.functional.cross_entropy(logits, torch.as_tensor(labels, dtype=torch.long))),
            "ece_15_bins": expected_calibration_error(probabilities, labels),
            "lane_change_share_error_pp": float(abs(np.mean(predictions < 2)-np.mean(labels < 2))*100),
            "class_priors": (np.bincount(labels, minlength=4)/len(labels)).tolist(),
            "classification_report": classification_report(labels, predictions, labels=range(4), target_names=CLASS_NAMES,
                                                             output_dict=True, zero_division=0),
            "confusion_counts": cm.tolist(), "confusion_row_normalized": normalized.tolist()}


def train_one(config, data, args, seed, run_dir, device):
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    # 打乱对照按车辆置换条件，仍保持每车标签固定；三集合独立置换。
    data = {key: {**part} for key, part in data.items()}
    if config.variant == "shuffled_preference":
        rng = np.random.default_rng(seed)
        for split_name, part in data.items():
            vehicle_lanes = part["meta"].groupby("_vehicle")["目的地车道"].first()
            shuffled = dict(zip(vehicle_lanes.index, rng.permutation(vehicle_lanes.to_numpy())))
            part["lane"] = part["meta"]["_vehicle"].map(shuffled).to_numpy(np.int64)
            save_json(run_dir / f"shuffled_{split_name}.json", {str(k): int(v) for k,v in shuffled.items()})
    train_loader = make_loader(data["train"], args.batch_size, device, True, args.balancing, seed)
    val_loader = make_loader(data["validation"], args.batch_size, device)
    model = EnhancedMultiVehicleLSTM(config).to(device)
    weights = None
    if args.balancing == "class_weighted":
        counts = np.bincount(data["train"]["y"], minlength=4)
        weights = torch.tensor(len(data["train"]["y"])/(4*counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        smooth_sum = nll_sum = loss_denominator_sum = 0.0
        total = correct = 0
        for x, style, lane, y in train_loader:
            x, style, lane, y = [v.to(device, non_blocking=True) for v in [x, style, lane, y]]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x, style, lane)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("训练损失非有限值，请检查输入和数值范围")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # 分类加权loss的reduction分母依赖类别权重，使用对应分母累计。
            denominator = float(weights[y].sum()) if weights is not None else len(y)
            smooth_sum += float(loss.detach())*denominator
            loss_denominator_sum += denominator
            nll_sum += float(nn.functional.cross_entropy(logits.detach(), y, reduction="sum"))
            total += len(y)
            correct += int((logits.argmax(1) == y).sum())
        val_logits, val_y = predict(model, val_loader, device)
        val_loss = float(nn.functional.cross_entropy(val_logits, val_y, label_smoothing=0.1,
                                                    weight=weights.cpu() if weights is not None else None))
        if not np.isfinite(val_loss):
            raise FloatingPointError("验证损失非有限值，不能据此选择模型")
        # 加权情形用实际累计分母，避免把末批当作整批平均。
        train_loss = smooth_sum/loss_denominator_sum
        row = {"epoch": epoch, "train_smoothed_ce": train_loss, "train_nll": nll_sum/total,
               "train_accuracy": correct/total, "validation_smoothed_ce": val_loss,
               "validation_nll": float(nn.functional.cross_entropy(val_logits, val_y)),
               "validation_accuracy": float((val_logits.argmax(1) == val_y).float().mean()),
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"seed={seed} epoch={epoch} train_CE={train_loss:.4f} val_CE={val_loss:.4f} "
              f"val_NLL={row['validation_nll']:.4f} val_ACC={row['validation_accuracy']:.4f}", flush=True)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    checkpoint = {"model_state_dict": best_state, "model_config": asdict(config), "features": FEATURES,
                  "class_names": CLASS_NAMES, "best_epoch": best_epoch, "best_validation_smoothed_ce": best_loss,
                  "seed": seed, "preprocessing": {"missing_distance": args.missing_distance,
                  "terminal_x": args.terminal_x, "no_absolute_position": args.no_absolute_position},
                  "balancing": args.balancing, "format_version": 2}
    # 旧版PyTorch在Windows中文路径下直接传Path可能失败，使用Python文件句柄。
    with (run_dir/"best_model.pth").open("wb") as handle:
        torch.save(checkpoint, handle)
    pd.DataFrame(history).to_csv(run_dir/"history.csv", index=False, encoding="utf-8-sig")
    # 只有最佳验证权重确定后才读取test loader，测试结果不参与早停。
    test_logits, test_y = predict(model, make_loader(data["test"], args.batch_size, device), device)
    result = metrics(test_logits, test_y)
    result.update({"seed": seed, "best_epoch": best_epoch,
                   "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    save_json(run_dir/"test_metrics.json", result)
    np.savez_compressed(run_dir/"test_predictions.npz", logits=test_logits.numpy(), labels=test_y.numpy(),
                        sample_ids=data["test"]["meta"]["_sample"].to_numpy(str))
    # 只使用训练自然先验与已知均衡采样先验进行后验修正，不拟合测试分布。
    if args.balancing == "undersample":
        prior = np.bincount(data["train"]["y"], minlength=4)/len(data["train"]["y"])
        corrected = test_logits + torch.tensor(np.log(prior/0.25), dtype=test_logits.dtype)
        save_json(run_dir/"test_metrics_prior_corrected.json", metrics(corrected, test_y))
    distance = (data["test"]["meta"]["目的地X"]-data["test"]["meta"]["质心X"]).to_numpy()
    strata = {}
    for threshold in [500, 300, 200, 100]:
        mask = distance > threshold
        strata[f"distance_gt_{threshold}m"] = metrics(test_logits[mask], test_y[mask]) if mask.any() else {"n": 0}
    save_json(run_dir/"distance_cumulative_thresholds.json", strata)
    return result


@torch.inference_mode()
def benchmark_forward(model, device, warmup=200, repeats=1000):
    """测量神经网络前向延迟，不包含控制器执行和SUMO交互。"""
    model.eval()
    c = model.config
    x = torch.zeros(1, c.time_steps, c.input_size, device=device)
    style = lane = torch.zeros(1, dtype=torch.long, device=device)
    for _ in range(warmup):
        model(x, style, lane)
    durations = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model(x, style, lane)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append((time.perf_counter()-start)*1000)
    return {"scope": "network_forward_only", "batch_size": 1, "precision": "FP32",
            "warmup": warmup, "repeats": repeats, "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)), "device": str(device)}


def load_model(checkpoint_path, device="cpu"):
    # 只载入自己生成且可信的checkpoint；不加载旧结构或未知pickle。
    with Path(checkpoint_path).open("rb") as handle:
        checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
    model = EnhancedMultiVehicleLSTM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def predict_windows(model, scaler, raw_frames, styles, preferred_lanes, preprocessing, device="cpu"):
    """闭环批量推理。raw_frames按车辆连续放置10帧，车辆顺序须与两个条件数组一致。

    只使用调用方已取得的历史状态；偏好由外部传入，不读历史CSV的终端标签。
    raw_frames含34个原始物理字段、时间戳及可选邻车ID，使用同一预处理/scaler。
    返回logits、四类概率、动作编号；本函数不实现IDM/SUMO安全执行器。
    """
    device = torch.device(device)
    model = model.to(device).eval()
    styles, preferred_lanes = np.asarray(styles), np.asarray(preferred_lanes)
    n = len(styles)
    if n < 1 or preferred_lanes.shape != (n,) or styles.shape != (n,) or len(raw_frames) != n*10:
        raise ValueError("输入必须为N个条件、N×10行历史状态")
    for name, values in [("驾驶风格", styles), ("外部偏好", preferred_lanes)]:
        if not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any((values<0)|(values>=3)):
            raise ValueError(f"{name}必须是0~2的整数")
    if preprocessing["terminal_x"] is None:
        destinations = pd.to_numeric(raw_frames["目的地X"], errors="raise").to_numpy()
        if not np.isfinite(destinations).all() or np.ptp(destinations)>1e-3:
            raise ValueError("闭环目的地X必须使用与训练对应的固定断面坐标")
    processed = preprocess_features(raw_frames, preprocessing["terminal_x"], preprocessing["missing_distance"])
    times = processed["时间戳"].to_numpy().reshape(n, 10)
    if not np.allclose(np.diff(times, axis=1), 0.1, atol=1e-4, rtol=0):
        raise ValueError("闭环历史状态必须按0.1秒递增，且不能跨车辆串接")
    X = scaler.transform(processed[FEATURES].to_numpy()).reshape(n, 10, 34).astype(np.float32)
    if preprocessing["no_absolute_position"]:
        X[:, :, [FEATURES.index("质心X"), FEATURES.index("目的地X")]] = 0.0
    logits, _ = model(torch.from_numpy(X).to(device), torch.as_tensor(styles, dtype=torch.long, device=device),
                      torch.as_tensor(preferred_lanes, dtype=torch.long, device=device))
    probabilities = logits.softmax(-1)
    return logits.cpu().numpy(), probabilities.cpu().numpy(), probabilities.argmax(-1).cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=SAVE_PATH)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--recording-col", help="不同录像重复车号时，必须提供真实录像来源列")
    parser.add_argument("--terminal-x", type=float, help="经核实的固定采样断面X坐标；不猜测实际坐标")
    parser.add_argument("--missing-distance", type=float, default=1000.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--balancing", choices=["undersample", "natural", "class_weighted"], default="undersample")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-absolute-position", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("epochs/batch-size必须大于0，训练种子不能重复")
    if not np.isfinite(args.missing_distance) or args.missing_distance <= 0:
        raise ValueError("missing-distance必须为正的有限数值")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, provenance = load_frames(args.data_dir, args.recording_col)
    audit = audit_frames(frames, args.terminal_x)
    audit["files"] = provenance
    save_json(args.output_dir/"data_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if audit["issues"]:
        raise ValueError("数据预检未通过，详见data_audit.json；请修复车道索引、车辆条件或目标坐标等异常后重试。")
    processed = preprocess_features(frames, args.terminal_x, args.missing_distance)
    del frames
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8")) if args.split_manifest else vehicle_split(processed, args.split_seed)
    validate_split(manifest, processed)
    data = {key: make_sequences(processed, ids) for key, ids in manifest.items()}
    del processed
    scaler = StandardScaler().fit(data["train"]["X"].reshape(-1, 34))
    removed_indices = [FEATURES.index("质心X"), FEATURES.index("目的地X")]
    for key, part in data.items():
        part["X"] = scaler.transform(part["X"].reshape(-1, 34)).reshape(-1, 10, 34).astype(np.float32)
        if args.no_absolute_position:
            # 标准化后置零，等价于移除绝对纵向坐标信息，保持输入宽度与参数量匹配。
            part["X"][:, :, removed_indices] = 0.0
    dataset_summary = {key: {"vehicles": len(manifest[key]), "windows": len(part["y"]),
                            "class_counts": np.bincount(part["y"], minlength=4).tolist()} for key, part in data.items()}
    save_json(args.output_dir/"dataset_summary.json", dataset_summary)
    save_json(args.output_dir/"vehicle_split.json", manifest)
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)
    if args.audit_only:
        print("数据预检通过；未训练。")
        return
    config = ModelConfig(variant=args.variant)
    tag = args.variant + ("_no_position" if args.no_absolute_position else "") + "_" + args.balancing
    experiment_dir = args.output_dir / tag
    # 不覆盖旧实验结果，避免不同设置混入同一汇总。
    experiment_dir.mkdir(exist_ok=False)
    joblib.dump(scaler, experiment_dir/"scaler.pkl")
    save_json(experiment_dir/"vehicle_split.json", manifest)
    save_json(experiment_dir/"features.json", FEATURES)
    save_json(experiment_dir/"data_audit.json", audit)
    save_json(experiment_dir/"dataset_summary.json", dataset_summary)
    save_json(experiment_dir/"run_config.json", {**{k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                                "model": asdict(config), "torch_version": torch.__version__,
                                                "numpy_version": np.__version__, "cuda_version": torch.version.cuda,
                                                "decision_label_provenance": "preconstructed_CSV_not_revalidated_here"})
    device = torch.device(args.device)
    results = [train_one(config, data, args, seed, experiment_dir/f"seed_{seed}", device) for seed in args.seeds]
    summary = {key: {"mean": float(np.mean([r[key] for r in results])),
                     "std": float(np.std([r[key] for r in results], ddof=1)) if len(results)>1 else None}
               for key in ["accuracy", "macro_f1", "nll", "ece_15_bins", "lane_change_share_error_pp"]}
    save_json(experiment_dir/"summary.json", {"seeds": args.seeds, "statistics": summary, "runs": results})
    if args.benchmark:
        model, _ = load_model(experiment_dir/f"seed_{args.seeds[0]}"/"best_model.pth", device)
        save_json(experiment_dir/"forward_latency.json", benchmark_forward(model, device))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
