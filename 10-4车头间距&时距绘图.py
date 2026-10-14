# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 读取数据
# df = pd.read_csv(r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\670预处理.csv")
#
# # 筛选有效数据
# valid = df[(df['前车ID'] != -1) & (df['车头时距'].between(0.1, 10))]
#
# # 按车辆计算平均值
# stats = valid.groupby('车辆ID').agg({
#     '前车间距': 'mean',
#     '车头时距': 'mean'
# }).rename(columns={'前车间距': '平均前车间距', '车头时距': '平均车头时距'}).reset_index()
#
# print(f"统计车辆数: {len(stats)}")
#
# # 绘制直方图+KDE
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#
# sns.distplot(stats['平均前车间距'].dropna(), bins=20,
#                  hist_kws={'color': 'lightblue', 'edgecolor': 'black', 'alpha': 0.7},
#                  kde_kws={'color': 'red', 'linewidth': 3}, ax=ax1)
#
# mean1 = stats['平均前车间距'].mean()
# ax1.axvline(mean1, color='gray', linestyle='--', linewidth=2, label=f"均值: {mean1:.1f}m")
# ax1.set_xlabel('平均车头间距 (m)', fontsize=15, fontweight='bold')
# ax1.set_ylabel('频率', fontsize=15, fontweight='bold')
#
# sns.distplot(stats['平均车头时距'].dropna(), bins=20,
#                  hist_kws={'color': 'lightcoral', 'edgecolor': 'black', 'alpha': 0.7},
#                  kde_kws={'color': 'blue', 'linewidth': 3}, ax=ax2)
#
# mean2 = stats['平均车头时距'].mean()
# ax2.axvline(mean2, color='gray', linestyle='--', linewidth=2, label=f"均值: {mean2:.1f}s")
# ax2.set_xlabel('平均车头时距 (s)', fontsize=15, fontweight='bold')
# ax2.set_ylabel('频率', fontsize=15, fontweight='bold')
#
# ax1.legend(fontsize=15)
# ax2.legend(fontsize=15)
# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
files = [
    r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845-1预处理.csv",
    r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845-2预处理.csv"
]

# 初始化统计数据列表
all_stats = []

# 分块处理每个文件
chunk_size = 100000  # 分块大小

for file_idx, file in enumerate(files, 1):
    print(f"处理文件{file_idx}...")

    for chunk in pd.read_csv(file, chunksize=chunk_size):
        # 筛选有效数据
        valid_chunk = chunk[(chunk['前车ID'] != -1) & (chunk['车头时距'].between(0.1, 10))]

        if not valid_chunk.empty:
            # 按车辆计算平均值
            chunk_stats = valid_chunk.groupby('车辆ID').agg({
                '前车间距': 'mean',
                '车头时距': 'mean'
            }).reset_index()

            all_stats.append(chunk_stats)

# 合并所有统计数据
if all_stats:
    all_stats_df = pd.concat(all_stats, ignore_index=True)

    # 再次按车辆ID分组，计算最终平均值
    stats = all_stats_df.groupby('车辆ID').agg({
        '前车间距': 'mean',
        '车头时距': 'mean'
    }).rename(columns={'前车间距': '平均前车间距', '车头时距': '平均车头时距'}).reset_index()

    print(f"统计车辆数: {len(stats)}")

    # 绘制直方图+KDE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    sns.distplot(stats['平均前车间距'].dropna(), bins=20,
                 hist_kws={'color': 'lightblue', 'edgecolor': 'black', 'alpha': 0.7},
                 kde_kws={'color': 'red', 'linewidth': 3}, ax=ax1)

    mean1 = stats['平均前车间距'].mean()
    ax1.axvline(mean1, color='gray', linestyle='--', linewidth=2, label=f"均值: {mean1:.1f}m")
    ax1.set_xlabel('平均车头间距 (m)', fontsize=15, fontweight='bold')
    ax1.set_ylabel('频率', fontsize=15, fontweight='bold')

    sns.distplot(stats['平均车头时距'].dropna(), bins=20,
                 hist_kws={'color': 'lightcoral', 'edgecolor': 'black', 'alpha': 0.7},
                 kde_kws={'color': 'blue', 'linewidth': 3}, ax=ax2)

    mean2 = stats['平均车头时距'].mean()
    ax2.axvline(mean2, color='gray', linestyle='--', linewidth=2, label=f"均值: {mean2:.1f}s")
    ax2.set_xlabel('平均车头时距 (s)', fontsize=15, fontweight='bold')
    ax2.set_ylabel('频率', fontsize=15, fontweight='bold')

    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    plt.tight_layout()
    plt.show()
else:
    print("没有有效数据")