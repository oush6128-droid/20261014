import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

categories = ['低流量\n(236辆)', '中流量\n(457辆)', '高流量\n(550辆)']
true_vals  = [199, 405, 472]
modelA_vals = [260, 566, 846]
modelB_vals = [192, 397, 429]

pct_A = [f'({(a-t)/t*100:+.1f}%)' for t, a in zip(true_vals, modelA_vals)]
pct_B = [f'({(b-t)/t*100:+.1f}%)' for t, b in zip(true_vals, modelB_vals)]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

# 颜色与密度-流量曲线一致：红=原始，蓝=A，绿=B
bars_true = ax.bar(x - width, true_vals, width, label='原始（真实轨迹）',
                   color='red', edgecolor='white')
bars_A = ax.bar(x, modelA_vals, width, label='Model-A（仅风格嵌入）',
                color='blue', edgecolor='white')
bars_B = ax.bar(x + width, modelB_vals, width, label='Model-B（风格+目的地嵌入）',
                color='green', edgecolor='white')

# 标注
for bar, val in zip(bars_true, true_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontsize=10)
for bar, val, pct in zip(bars_A, modelA_vals, pct_A):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontsize=10)
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 25,
            pct, ha='center', va='bottom', fontsize=9, color='black')
for bar, val, pct in zip(bars_B, modelB_vals, pct_B):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontsize=10)
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 25,
            pct, ha='center', va='bottom', fontsize=9, color='black')

ax.set_xlabel('流量级别')
ax.set_ylabel('生成车辆数（辆）')
ax.set_title('不同流量级别下模型生成车辆数与真实观测值对比')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1000)

plt.tight_layout()
plt.show()