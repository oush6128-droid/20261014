import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel(r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845断面分析统计.xlsx", sheet_name='统计')
points = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
periods = df['时间段min']

# 提取车辆数
data = {}
for p in points:
    col = f'{p}米断面车辆数'
    if col in df.columns:
        data[p] = df[col].values

# 绘图
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(points))

colors = plt.cm.viridis([0.1, 0.3, 0.5, 0.7, 0.85, 0.95])

bottom = np.zeros(len(points))

for i, period in enumerate(periods):
    counts = [data[p][i] for p in points]
    ax.bar(x, counts, 0.6, bottom=bottom, label=period,
           color=colors[i], edgecolor='white', linewidth=0.8)
    bottom += counts

# 标注总数
for i, p in enumerate(points):
    total = sum(data[p])
    ax.text(i, total + 0.5, str(int(total)), ha='center', va='bottom',
            fontweight='bold', fontsize=9)

# 设置坐标轴
ax.set_xticks(x)
ax.set_xticklabels([f'{p}米' for p in points], fontsize=10)
ax.set_xlabel('断面位置', fontsize=15, fontweight='bold')
ax.set_ylabel('车辆数', fontsize=15, fontweight='bold')

# 图例放在图形右侧外部
ax.legend(title='时间段(分钟)', fontsize=10, title_fontsize=12,
          loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

plt.tight_layout()
plt.show()