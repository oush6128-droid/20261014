import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel(r"D:\桌面\车载组\6-4_徐东航拍数据\第一次\932断面分析统计.xlsx", sheet_name='统计')
points = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
periods = df['时间段min']

# 提取速度
speeds = {}
for p in points:
    col = f'{p}米断面速度'
    if col in df.columns:
        speeds[p] = df[col].values

# 绘图
fig, ax = plt.subplots(figsize=(13, 7))
x = np.arange(len(points))

# 6种高对比度颜色
colors = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b'  # 棕色
]

# 标记样式
markers = ['o', 's', 'D', '^', 'v', 'p']
line_styles = ['-', '--', '-.', ':', '-.', '--']

for i, period in enumerate(periods):
    # 获取当前时间段各断面速度
    period_speeds = [speeds[p][i] for p in points]

    # 绘制折线
    ax.plot(x, period_speeds,
            color=colors[i],
            marker=markers[i],
            linestyle=line_styles[i],
            linewidth=2.5,
            markersize=9,
            label=period,
            markeredgecolor='white',
            markeredgewidth=1.2,
            alpha=0.9)

# 设置x轴
ax.set_xticks(x)
ax.set_xticklabels([f'{p}米' for p in points], fontsize=10, rotation=0)
ax.set_xlabel('断面位置', fontsize=15, fontweight='bold')
ax.set_ylabel('平均速度 (m/s)', fontsize=15, fontweight='bold')

# 图例
ax.legend(title='时间段(分钟)', fontsize=12, title_fontsize=14,
          loc='best', frameon=True, framealpha=0.9)

plt.tight_layout()
plt.show()