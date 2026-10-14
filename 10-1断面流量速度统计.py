# import pandas as pd
# import numpy as np
# from tqdm import tqdm
#
# # 读取数据
# df = pd.read_csv(r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\670预处理.csv")
# points = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
#
# # 断面分析
# results = []
# for vid, vdf in tqdm(df.groupby('车辆ID'), desc="断面分析"):
#     vdf = vdf.sort_values('时间戳')
#     res = {'车辆ID': vid}
#
#     for p in points:
#         time_val = speed_val = None
#         x_vals = vdf['质心X'].values
#
#         # 精确匹配
#         exact = np.where(x_vals == p)[0]
#         if len(exact) > 0:
#             idx = exact[0]
#             time_val, speed_val = vdf.iloc[idx]['时间戳'], vdf.iloc[idx]['纵向速度']
#         else:
#             # 跨越匹配
#             for i in range(len(x_vals) - 1):
#                 x1, x2 = x_vals[i], x_vals[i + 1]
#                 if (x1 < p < x2) or (x1 > p > x2):
#                     row1, row2 = vdf.iloc[i], vdf.iloc[i + 1]
#                     r = (p - x1) / (x2 - x1) if x2 != x1 else 0.5
#                     time_val = row1['时间戳'] + r * (row2['时间戳'] - row1['时间戳'])
#                     speed_val = row1['纵向速度'] + r * (row2['纵向速度'] - row1['纵向速度'])
#                     break
#
#         res[f'{p}米时间'], res[f'{p}米速度'] = time_val, speed_val
#
#     results.append(res)
#
# result_df = pd.DataFrame(results)
# cols = ['车辆ID'] + [f'{p}米{col}' for p in points for col in ['时间', '速度']]
# result_df = result_df[cols]
#
# # 统计（固定时间段）
# time_ranges = [
#     (0, 300, "0-5"),  # 0-5分钟
#     (300, 600, "5-10"),  # 5-10分钟
#     (600, 900, "10-15"),  # 10-15分钟
#     (900, 1200, "15-20"),  # 15-20分钟
#     (1200, 1500, "20-25"),  # 20-25分钟
#     (1500, 1800, "25-30")  # 25-30分钟
# ]
#
# # 创建统计表
# stats = []
# for start, end, label in time_ranges:
#     row = {'时间段min': label}
#
#     for p in points:
#         t_col, s_col = f'{p}米时间', f'{p}米速度'
#
#         # 筛选时间段内的车辆
#         mask = (result_df[t_col] > start) & (result_df[t_col] <= end)
#         filtered = result_df[mask]
#
#         count = len(filtered)
#         speeds = filtered[s_col].dropna()
#         avg_speed = speeds.mean() if not speeds.empty else np.nan
#
#         row[f'{p}米断面_车辆数'] = count
#         row[f'{p}米断面_速度'] = round(avg_speed, 2) if not pd.isna(avg_speed) else np.nan
#
#     stats.append(row)
#
# stats_df = pd.DataFrame(stats)
#
# # 重排列
# cols_order = ['时间段min']
# for p in points:
#     cols_order.extend([f'{p}米断面_车辆数', f'{p}米断面_速度'])
# stats_df = stats_df[cols_order]
#
# # 重命名
# stats_df.columns = ['时间段min'] + [f'{p}米断面车辆数' if '车辆数' in c else f'{p}米断面速度'
#                                     for p in points for c in ['车辆数', '速度']]
#
# # 保存
# output = r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\670断面分析统计.xlsx"
# with pd.ExcelWriter(output, engine='openpyxl') as writer:
#     result_df.to_excel(writer, sheet_name='断面分析', index=False)
#     stats_df.to_excel(writer, sheet_name='统计', index=False)
#
# print(f"完成！保存到: {output}")


# import pandas as pd
# import numpy as np
# from tqdm import tqdm
#
# # 文件路径
# files = [
#     r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845-1预处理.csv",
#     r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845-2预处理.csv"
# ]
# output = r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845断面分析统计.xlsx"
#
# points = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
# time_ranges = [
#     (0, 300, "0-5"), (300, 600, "5-10"), (600, 900, "10-15"),
#     (900, 1200, "15-20"), (1200, 1500, "20-25"), (1500, 1800, "25-30")
# ]
#
#
# # 断面分析函数
# def process_vehicle_data(vdf, points):
#     res = {'车辆ID': vdf['车辆ID'].iloc[0]}
#     vdf = vdf.sort_values('时间戳')
#     x_vals = vdf['质心X'].values
#
#     for p in points:
#         time_val = speed_val = None
#         exact = np.where(x_vals == p)[0]
#
#         if len(exact) > 0:
#             idx = exact[0]
#             time_val, speed_val = vdf.iloc[idx]['时间戳'], vdf.iloc[idx]['纵向速度']
#         else:
#             for i in range(len(x_vals) - 1):
#                 x1, x2 = x_vals[i], x_vals[i + 1]
#                 if (x1 < p < x2) or (x1 > p > x2):
#                     row1, row2 = vdf.iloc[i], vdf.iloc[i + 1]
#                     r = (p - x1) / (x2 - x1) if x2 != x1 else 0.5
#                     time_val = row1['时间戳'] + r * (row2['时间戳'] - row1['时间戳'])
#                     speed_val = row1['纵向速度'] + r * (row2['纵向速度'] - row1['纵向速度'])
#                     break
#
#         res[f'{p}米时间'], res[f'{p}米速度'] = time_val, speed_val
#
#     return res
#
#
# # 分块处理
# all_results = []
# chunk_size = 50000
#
# for file_idx, file in enumerate(files, 1):
#     print(f"处理文件{file_idx}...")
#     file_results = []
#
#     for chunk in tqdm(pd.read_csv(file, chunksize=chunk_size), desc=f"文件{file_idx}"):
#         for vid, vdf in chunk.groupby('车辆ID'):
#             file_results.append(process_vehicle_data(vdf, points))
#
#     all_results.extend(file_results)
#
# # 创建结果DataFrame
# result_df = pd.DataFrame(all_results)
# cols = ['车辆ID'] + [f'{p}米{col}' for p in points for col in ['时间', '速度']]
# result_df = result_df[cols]
#
# # 统计
# stats = []
# for start, end, label in time_ranges:
#     row = {'时间段min': label}
#
#     for p in points:
#         t_col, s_col = f'{p}米时间', f'{p}米速度'
#         mask = (result_df[t_col] > start) & (result_df[t_col] <= end)
#         filtered = result_df[mask]
#
#         count = len(filtered)
#         speeds = filtered[s_col].dropna()
#         avg_speed = speeds.mean() if not speeds.empty else np.nan
#
#         row[f'{p}米断面_车辆数'] = count
#         row[f'{p}米断面_速度'] = round(avg_speed, 2) if not pd.isna(avg_speed) else np.nan
#
#     stats.append(row)
#
# stats_df = pd.DataFrame(stats)
# cols_order = ['时间段min'] + [f'{p}米断面_{col}' for p in points for col in ['车辆数', '速度']]
# stats_df = stats_df[cols_order]
#
# # 重命名
# stats_df.columns = ['时间段min'] + [f'{p}米断面{col}' for p in points for col in ['车辆数', '速度']]
#
# # 保存
# with pd.ExcelWriter(output, engine='openpyxl') as writer:
#     result_df.to_excel(writer, sheet_name='断面分析', index=False)
#     stats_df.to_excel(writer, sheet_name='统计', index=False)
#
# print(f"完成！保存到: {output}, 车辆数: {len(result_df)}")