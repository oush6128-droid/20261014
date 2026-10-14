import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 文件路径
excel_path = r"D:\桌面\车载组\6-4_徐东航拍数据\第一次\932断面分析统计.xlsx"

# 读取数据
print("正在读取Excel文件...")
# 读取空间占有率数据
spatial_df = pd.read_excel(excel_path, sheet_name='空间占有率')
# 读取时间占有率数据
time_df = pd.read_excel(excel_path, sheet_name='时间占有率')

# 创建5分钟时间区间
time_bins = [0, 300, 600, 900, 1200, 1500]
time_labels = ['0-5', '5-10', '10-15', '15-20', '20-25']

# 处理空间占有率数据
spatial_df['时间区间'] = pd.cut(spatial_df['时间戳'], bins=time_bins, labels=time_labels, right=False, include_lowest=True)
spatial_avg = spatial_df.groupby('时间区间')[['0车道', '1车道', '2车道']].mean().reset_index()

# 创建包含两个子图的大图，增加宽度，调整间距
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
plt.subplots_adjust(wspace=1.05, bottom=0.15)  # 增加子图间距，为x轴标签留出空间

# 设置浅色柱体颜色
colors = ['lightcoral', 'lightgreen', 'lightblue']

# 1. 绘制空间占有率5分钟平均柱形图
ax1 = axes[0]

# 设置柱形图参数
x = np.arange(len(time_labels))
width = 0.25  # 柱体宽度

# 为每个车道绘制柱体
bars1 = ax1.bar(x - width, spatial_avg['0车道'], width,
                color=colors[0], edgecolor='darkred', linewidth=1.2, alpha=0.85)
bars2 = ax1.bar(x, spatial_avg['1车道'], width,
                color=colors[1], edgecolor='darkgreen', linewidth=1.2, alpha=0.85)
bars3 = ax1.bar(x + width, spatial_avg['2车道'], width,
                color=colors[2], edgecolor='darkblue', linewidth=1.2, alpha=0.85)

# 设置图表属性
ax1.set_xlabel('时间段 (分钟)', fontsize=13, fontweight='bold')
ax1.set_ylabel('空间占有率 (%)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(time_labels)

# 设置y轴范围
max_val_spatial = spatial_avg[['0车道', '1车道', '2车道']].values.max()
ax1.set_ylim(0, max_val_spatial * 1.03)

# 2. 绘制时间占有率柱形图
ax2 = axes[1]

# 设置柱形图参数
x_time = np.arange(len(time_df))
width_time = 0.25  # 柱体宽度

# 为每个车道绘制柱体
bars_time1 = ax2.bar(x_time - width_time, time_df['0车道'], width_time,
                     color=colors[0], edgecolor='darkred', linewidth=1.2, alpha=0.85)
bars_time2 = ax2.bar(x_time, time_df['1车道'], width_time,
                     color=colors[1], edgecolor='darkgreen', linewidth=1.2, alpha=0.85)
bars_time3 = ax2.bar(x_time + width_time, time_df['2车道'], width_time,
                     color=colors[2], edgecolor='darkblue', linewidth=1.2, alpha=0.85)

# 设置图表属性
ax2.set_xlabel('检测断面位置 (米)', fontsize=13, fontweight='bold')
ax2.set_ylabel('时间占有率 (%)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_time)
ax2.set_xticklabels([f"{int(x)}" for x in time_df['断面']])

# 设置y轴范围
max_val_time = time_df[['0车道', '1车道', '2车道']].values.max()
ax2.set_ylim(0, max_val_time * 1.03)

# 调整x轴刻度标签位置
ax2.tick_params(axis='x', which='major', pad=3)

# 添加一个共享的图例
# 创建自定义图例元素
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors[0], edgecolor='darkred', alpha=0.85, label='0车道'),
    Patch(facecolor=colors[1], edgecolor='darkgreen', alpha=0.85, label='1车道'),
    Patch(facecolor=colors[2], edgecolor='darkblue', alpha=0.85, label='2车道')
]

# 在图表顶部居中位置添加图例
fig.legend(handles=legend_elements, loc='upper center',
           bbox_to_anchor=(0.5, 1), ncol=3, fontsize=12,
           frameon=True, fancybox=True, shadow=False, framealpha=0.9)

# 显示图表
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 输出统计信息
print("统计摘要:")
print("\n空间占有率统计:")
for lane, name in zip(['0车道', '1车道', '2车道'], ['0车道', '1车道', '2车道']):
    avg_val = spatial_avg[lane].mean()
    print(f"  {name} 平均空间占有率: {avg_val:.2f}%")

print("\n时间占有率统计:")
for lane, name in zip(['0车道', '1车道', '2车道'], ['0车道', '1车道', '2车道']):
    avg_val = time_df[lane].mean()
    print(f"  {name} 平均时间占有率: {avg_val:.4f}%")
print("图表绘制完成！")
