import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ===================== 路径与参数 =====================
INPUT_FILE = r"C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\IDM  跟驰\ZUI最终合并数据.csv"
OUTPUT_DIR = os.path.dirname(INPUT_FILE)
DT = 0.1
MIN_SEG_LEN = 10
MAX_SEGMENTS = None  # 每个风格最多标定段数，None=全部

# ===================== IDM 模型 (标量版本, 用于仿真循环) =====================
def IDM(speed, speed_lead, gap, a_max, v_des, T_des, s0, b_comf, beta):
    delta_v = speed - speed_lead
    sqrt_ab = (a_max * b_comf + 1e-8) ** 0.5
    s_star = s0 + speed * T_des + (speed * delta_v) / (2.0 * sqrt_ab)
    if s_star < s0:
        s_star = s0
    acc_free = a_max * (1.0 - (speed / (v_des + 1e-8)) ** beta)
    acc_inter = a_max * (s_star / (gap + 1e-8)) ** 2.0
    return acc_free - acc_inter, s_star


def IDM_free(speed, a_max, v_des, beta):
    return a_max * (1.0 - (speed / (v_des + 1e-8)) ** beta)


def rmspe(y_true, y_pred, eps=0.1):
    denom = np.maximum(np.abs(y_true), eps)
    return np.sqrt(np.mean(np.square((y_true - y_pred) / denom)))


# ===================== 多步轨迹仿真 =====================
def simulate_follow(v0, v_lead, gap0, a_max, v_des, T_des, s0, b_comf, beta, n_steps):
    """多步跟驰仿真：从初始状态滚动模拟整条轨迹"""
    v_sim = np.empty(n_steps + 1)
    gap_sim = np.empty(n_steps + 1)
    v_sim[0] = v0
    gap_sim[0] = gap0
    for t in range(n_steps):
        acc, _ = IDM(v_sim[t], v_lead[t], gap_sim[t],
                     a_max, v_des, T_des, s0, b_comf, beta)
        v_new = v_sim[t] + acc * DT
        v_sim[t + 1] = 0.1 if v_new < 0.1 else (60.0 if v_new > 60.0 else v_new)
        g_new = gap_sim[t] + (v_lead[t] - v_sim[t]) * DT
        gap_sim[t + 1] = 0.1 if g_new < 0.1 else (9999.0 if g_new > 9999.0 else g_new)
    return v_sim, gap_sim


def simulate_free(v0, a_max, v_des, beta, n_steps):
    """多步自由行驶仿真"""
    v_sim = np.empty(n_steps + 1)
    v_sim[0] = v0
    for t in range(n_steps):
        acc = IDM_free(v_sim[t], a_max, v_des, beta)
        v_new = v_sim[t] + acc * DT
        v_sim[t + 1] = 0.1 if v_new < 0.1 else (60.0 if v_new > 60.0 else v_new)
    return v_sim


# ===================== 单段标定 =====================
def calibrate_segment_follow(seg_df):
    """对单个跟驰片段进行IDM标定，段内7:3划分为标定集/验证集"""
    seg_df = seg_df.sort_index()
    v_all = seg_df["speed_x"].values
    v_lead_all = seg_df["speed_lead"].values
    gap_all = seg_df["gap"].values
    n = len(v_all)

    if n < MIN_SEG_LEN + 1:
        return None

    # 段内 7:3 train/test split
    n_train = int(n * 0.7)
    v, v_lead, gap = v_all[:n_train], v_lead_all[:n_train], gap_all[:n_train]
    v_t, v_lead_t, gap_t = v_all[n_train:], v_lead_all[n_train:], gap_all[n_train:]

    if len(v) < 5 or len(v_t) < 3:
        return None

    def target(x):
        a_max, v_des, T_des, s0, b_comf, beta = x
        n_steps = len(v) - 1
        v_sim, gap_sim = simulate_follow(
            v[0], v_lead, gap[0], a_max, v_des, T_des, s0, b_comf, beta, n_steps)
        ev = rmspe(v[1:], v_sim[1:])
        es = rmspe(gap[1:], gap_sim[1:])
        val = ev + es
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val

    bounds = [
        (0.1, 5.0),     # a_max
        (10.0, 25.0),   # v_des (论文表4-3边界)
        (0.1, 5.0),     # T_des
        (0.1, 10.0),    # s0
        (0.1, 5.0),     # b_comf
        (3.0, 5.0),     # beta
    ]

    result = differential_evolution(
        target, bounds, seed=42, maxiter=30, polish=False,
        popsize=6, mutation=(0.5, 1.5), recombination=0.9,
        tol=1e-4, atol=0
    )
    p = {"a_max": result.x[0], "v_des": result.x[1], "T_des": result.x[2],
         "s0": result.x[3], "b_comf": result.x[4], "beta": result.x[5]}

    # 训练集误差
    n_s = len(v) - 1
    v_sim, gap_sim = simulate_follow(v[0], v_lead, gap[0], **p, n_steps=n_s)
    ev_tr = rmspe(v[1:], v_sim[1:])
    es_tr = rmspe(gap[1:], gap_sim[1:])

    # 测试集误差
    n_s_t = len(v_t) - 1
    v_sim_t, gap_sim_t = simulate_follow(v_t[0], v_lead_t, gap_t[0], **p, n_steps=n_s_t)
    ev_te = rmspe(v_t[1:], v_sim_t[1:])
    es_te = rmspe(gap_t[1:], gap_sim_t[1:])

    return {**p,
            "ev_train": ev_tr, "es_train": es_tr, "total_train": ev_tr + es_tr,
            "ev_test": ev_te, "es_test": es_te, "total_test": ev_te + es_te,
            "n_train": n_s, "n_test": n_s_t,
            "车辆ID": seg_df["车辆ID"].iloc[0],
            "状态": seg_df["状态"].iloc[0],
            "驾驶风格": int(seg_df["驾驶风格"].iloc[0])}


def calibrate_segment_free(seg_df):
    """对单个自由行驶片段进行IDM标定"""
    seg_df = seg_df.sort_index()
    v_all = seg_df["speed_x"].values
    n = len(v_all)

    if n < MIN_SEG_LEN + 1:
        return None

    n_train = int(n * 0.7)
    v = v_all[:n_train]
    v_t = v_all[n_train:]

    if len(v) < 5 or len(v_t) < 3:
        return None

    def target(x):
        a_max, v_des, beta = x
        n_steps = len(v) - 1
        v_sim = simulate_free(v[0], a_max, v_des, beta, n_steps)
        val = rmspe(v[1:], v_sim[1:])
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val

    bounds = [
        (0.1, 5.0),     # a_max
        (10.0, 25.0),   # v_des (论文表4-3边界)
        (3.0, 5.0),     # beta
    ]

    result = differential_evolution(
        target, bounds, seed=42, maxiter=25, polish=False,
        popsize=5, mutation=(0.5, 1.5), recombination=0.9,
        tol=1e-4, atol=0
    )
    p = {"a_max": result.x[0], "v_des": result.x[1], "beta": result.x[2]}

    n_s = len(v) - 1
    v_sim = simulate_free(v[0], **p, n_steps=n_s)
    ev_tr = rmspe(v[1:], v_sim[1:])

    n_s_t = len(v_t) - 1
    v_sim_t = simulate_free(v_t[0], **p, n_steps=n_s_t)
    ev_te = rmspe(v_t[1:], v_sim_t[1:])

    return {**p,
            "ev_train": ev_tr, "ev_test": ev_te,
            "n_train": n_s, "n_test": n_s_t,
            "车辆ID": seg_df["车辆ID"].iloc[0],
            "状态": seg_df["状态"].iloc[0],
            "驾驶风格": int(seg_df["驾驶风格"].iloc[0])}


# ===================== 主程序 =====================
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("  IDM标定 — 逐段标定 + 多步轨迹仿真 + 差分进化")
    print("  论文方法: 每(车辆ID,状态)片段独立标定, 按驾驶风格聚合")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    # 收集所有(车辆ID, 状态)分组
    groups = list(df.groupby(["车辆ID", "状态"]))

    follow_results = []
    free_results = []

    # 按驾驶风格分组段
    seg_sty_follow = {1: [], 2: [], 3: []}
    seg_sty_free = {1: [], 2: [], 3: []}

    print("\n正在收集片段...")
    for (vid, seg), g in groups:
        sty = int(g["驾驶风格"].iloc[0])
        if g["is_following"].iloc[0] == 1:
            seg_sty_follow[sty].append(g)
        else:
            seg_sty_free[sty].append(g)

    total_follow = sum(len(v) for v in seg_sty_follow.values())
    total_free = sum(len(v) for v in seg_sty_free.values())
    for sty in [1, 2, 3]:
        print(f"  风格{sty}: 跟驰{len(seg_sty_follow[sty])}段, 自由{len(seg_sty_free[sty])}段")
    print(f"  总计: 跟驰{total_follow}段, 自由{total_free}段")

    # ===== 跟驰标定 =====
    for sty in [1, 2, 3]:
        segments = seg_sty_follow[sty]
        if MAX_SEGMENTS and len(segments) > MAX_SEGMENTS:
            segments = segments[:MAX_SEGMENTS]

        n_seg = len(segments)
        print(f"\n{'='*50}")
        print(f"  风格{sty} 跟驰标定 ({n_seg}段)")
        print(f"{'='*50}")

        t0 = time.time()
        for i, g in enumerate(segments):
            res = calibrate_segment_follow(g)
            if res is not None:
                follow_results.append(res)
            if (i + 1) % 100 == 0 or i == n_seg - 1:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_seg - i - 1)
                print(f"  [{i+1}/{n_seg}] {elapsed:.0f}s ETA:{eta:.0f}s ", end="")
                if res:
                    print(f"a={res['a_max']:.3f} T={res['T_des']:.3f} "
                          f"s0={res['s0']:.3f} b={res['b_comf']:.3f} "
                          f"tr={res['total_train']:.4f} te={res['total_test']:.4f}")
                else:
                    print("跳过")

    # ===== 自由标定 =====
    for sty in [1, 2, 3]:
        segments = seg_sty_free[sty]
        if MAX_SEGMENTS and len(segments) > MAX_SEGMENTS:
            segments = segments[:MAX_SEGMENTS]

        n_seg = len(segments)
        print(f"\n{'='*50}")
        print(f"  风格{sty} 自由标定 ({n_seg}段)")
        print(f"{'='*50}")

        t0 = time.time()
        for i, g in enumerate(segments):
            res = calibrate_segment_free(g)
            if res is not None:
                free_results.append(res)
            if (i + 1) % 50 == 0 or i == n_seg - 1:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_seg - i - 1)
                print(f"  [{i+1}/{n_seg}] {elapsed:.0f}s ETA:{eta:.0f}s ", end="")
                if res:
                    print(f"a={res['a_max']:.3f} v0={res['v_des']:.3f} "
                          f"b={res['beta']:.3f} tr={res['ev_train']:.4f} te={res['ev_test']:.4f}")
                else:
                    print("跳过")

    # ===== 聚合结果 =====
    df_f = pd.DataFrame(follow_results)
    df_fr = pd.DataFrame(free_results)

    if len(df_f) == 0:
        print("\n错误: 没有成功标定的跟驰段!")
        exit()

    param_cols = ["a_max", "v_des", "T_des", "s0", "b_comf", "beta"]
    follow_params = {}
    free_params = {}

    print(f"\n{'='*70}")
    print("  标定结果 — 按驾驶风格聚合 (论文表4-5)")
    print(f"{'='*70}")

    for sty in [1, 2, 3]:
        sub = df_f[df_f["驾驶风格"] == sty]
        if len(sub) > 0:
            means = {c: round(sub[c].mean(), 4) for c in param_cols}
            follow_params[str(sty)] = means
            print(f"\n  风格{sty} 跟驰 ({len(sub)}段):")
            print(f"    a_max={means['a_max']:.4f} v_des={means['v_des']:.4f} "
                  f"T_des={means['T_des']:.4f} s0={means['s0']:.4f} "
                  f"b_comf={means['b_comf']:.4f} beta={means['beta']:.4f}")

    for sty in [1, 2, 3]:
        sub = df_fr[df_fr["驾驶风格"] == sty]
        if len(sub) > 0:
            means = {c: round(sub[c].mean(), 4) for c in ["a_max", "v_des", "beta"]}
            free_params[str(sty)] = means
            print(f"\n  风格{sty} 自由 ({len(sub)}段):")
            print(f"    a_max={means['a_max']:.4f} v_des={means['v_des']:.4f} "
                  f"beta={means['beta']:.4f}")

    # ===== 保存参数 =====
    param_path = os.path.join(OUTPUT_DIR, "IDM最终参数_逐段标定.json")
    with open(param_path, 'w', encoding='utf-8') as f:
        json.dump({"follow": follow_params, "free": free_params}, f,
                  indent=4, ensure_ascii=False)

    # ===== 保存逐段标定详情 =====
    detail_cols = ["车辆ID", "状态", "驾驶风格",
                   "a_max", "v_des", "T_des", "s0", "b_comf", "beta",
                   "ev_train", "es_train", "total_train",
                   "ev_test", "es_test", "total_test"]
    df_f[detail_cols].to_csv(
        os.path.join(OUTPUT_DIR, "IDM跟驰逐段标定结果.csv"),
        index=False, encoding="utf-8-sig")

    free_detail_cols = ["车辆ID", "状态", "驾驶风格",
                        "a_max", "v_des", "beta",
                        "ev_train", "ev_test"]
    df_fr[free_detail_cols].to_csv(
        os.path.join(OUTPUT_DIR, "IDM自由逐段标定结果.csv"),
        index=False, encoding="utf-8-sig")

    # ===== 误差汇总 (论文表4-4) =====
    print(f"\n{'='*70}")
    print("  误差汇总 (论文表4-4)")
    print(f"{'='*70}")

    for sty in [1, 2, 3]:
        sub = df_f[df_f["驾驶风格"] == sty]
        if len(sub) > 0:
            print(f"\n  风格{sty} 跟驰 ({len(sub)}段):")
            print(f"    速度RMSPE: 标定集={sub['ev_train'].mean():.4f}±{sub['ev_train'].std():.4f}  "
                  f"验证集={sub['ev_test'].mean():.4f}±{sub['ev_test'].std():.4f}")
            print(f"    间距RMSPE: 标定集={sub['es_train'].mean():.4f}±{sub['es_train'].std():.4f}  "
                  f"验证集={sub['es_test'].mean():.4f}±{sub['es_test'].std():.4f}")
            print(f"    综合RMSPE: 标定集={sub['total_train'].mean():.4f}±{sub['total_train'].std():.4f}  "
                  f"验证集={sub['total_test'].mean():.4f}±{sub['total_test'].std():.4f}")

    for sty in [1, 2, 3]:
        sub = df_fr[df_fr["驾驶风格"] == sty]
        if len(sub) > 0:
            print(f"\n  风格{sty} 自由 ({len(sub)}段):")
            print(f"    速度RMSPE: 标定集={sub['ev_train'].mean():.4f}±{sub['ev_train'].std():.4f}  "
                  f"验证集={sub['ev_test'].mean():.4f}±{sub['ev_test'].std():.4f}")

    print(f"\n  全部跟驰 ({len(df_f)}段):")
    print(f"    速度RMSPE: 标定集={df_f['ev_train'].mean():.4f}±{df_f['ev_train'].std():.4f}  "
          f"验证集={df_f['ev_test'].mean():.4f}±{df_f['ev_test'].std():.4f}")
    print(f"    间距RMSPE: 标定集={df_f['es_train'].mean():.4f}±{df_f['es_train'].std():.4f}  "
          f"验证集={df_f['es_test'].mean():.4f}±{df_f['es_test'].std():.4f}")
    print(f"    综合RMSPE: 标定集={df_f['total_train'].mean():.4f}±{df_f['total_train'].std():.4f}  "
          f"验证集={df_f['total_test'].mean():.4f}±{df_f['total_test'].std():.4f}")

    print(f"\n  全部自由 ({len(df_fr)}段):")
    print(f"    速度RMSPE: 标定集={df_fr['ev_train'].mean():.4f}±{df_fr['ev_train'].std():.4f}  "
          f"验证集={df_fr['ev_test'].mean():.4f}±{df_fr['ev_test'].std():.4f}")

    total_time = time.time() - t_start
    print(f"\n总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"标定完成! 参数保存至: {param_path}")
    print(f"逐段结果: IDM跟驰逐段标定结果.csv / IDM自由逐段标定结果.csv")
