import pandas as pd
import numpy as np
from tqdm import tqdm

# 读取数据
df = pd.read_csv(r"C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\1845-2处理3加速度调车道.csv")
output_path = r"C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\1845-2全.csv"


# 步骤(2): 确定车道
def get_lane(y):
    if 0 < y < 3.5:
        return 0
    elif 3.5 < y < 7:
        return 1
    elif 7 < y < 10.5:
        return 2
    return -1


df['车道'] = df['质心Y'].apply(get_lane)


# 步骤(4): 计算航向角
def calculate_heading(group):
    x = group['质心X'].values
    y = group['质心Y'].values
    heading = np.zeros(len(group))

    for i in range(1, len(group)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        if dx != 0:
            heading[i] = np.arctan2(dy, dx)
        else:
            heading[i] = np.pi / 2 if dy > 0 else -np.pi / 2

    if len(group) > 1:
        heading[0] = heading[1]

    group['航向角'] = heading
    return group


df = df.groupby('车辆ID', group_keys=False).apply(calculate_heading)

# 步骤(3): 寻找周围车辆
# 保存结果列表
result_rows = []

# 获取所有时间戳
all_timestamps = sorted(df['时间戳'].unique())

print("正在寻找周围车辆...")
for ts in tqdm(all_timestamps, desc="处理时间戳"):
    # 当前时刻的所有车辆
    current_vehicles = df[df['时间戳'] == ts].copy()

    for _, ego in current_vehicles.iterrows():
        ego_id = ego['车辆ID']
        ego_lane = ego['车道']
        ego_x = ego['质心X']

        # 初始化周围车辆信息
        surr_info = {
            '前车ID': -1, '前车间距': np.nan, '前车纵向速度': np.nan, '前车纵向加速度': np.nan,
            '后车ID': -1, '后车间距': np.nan, '后车纵向速度': np.nan, '后车纵向加速度': np.nan,
            '左前车ID': -1, '左前车间距': np.nan, '左前车纵向速度': np.nan, '左前车纵向加速度': np.nan,
            '左后车ID': -1, '左后车间距': np.nan, '左后车纵向速度': np.nan, '左后车纵向加速度': np.nan,
            '右前车ID': -1, '右前车间距': np.nan, '右前车纵向速度': np.nan, '右前车纵向加速度': np.nan,
            '右后车ID': -1, '右后车间距': np.nan, '右后车纵向速度': np.nan, '右后车纵向加速度': np.nan
        }

        # 筛选同车道车辆
        same_lane = current_vehicles[current_vehicles['车道'] == ego_lane]
        same_lane = same_lane[same_lane['车辆ID'] != ego_id]

        # 前车
        front_cars = same_lane[same_lane['质心X'] > ego_x]
        if not front_cars.empty:
            front = front_cars.loc[front_cars['质心X'].idxmin()]  # 最近的
            spacing = front['质心X'] - ego_x
            surr_info.update({
                '前车ID': front['车辆ID'], '前车间距': spacing,
                '前车纵向速度': front['纵向速度'], '前车纵向加速度': front['纵向加速度']
            })

        # 后车
        rear_cars = same_lane[same_lane['质心X'] < ego_x]
        if not rear_cars.empty:
            rear = rear_cars.loc[rear_cars['质心X'].idxmax()]
            spacing = ego_x - rear['质心X']
            surr_info.update({
                '后车ID': rear['车辆ID'], '后车间距': spacing,
                '后车纵向速度': rear['纵向速度'], '后车纵向加速度': rear['纵向加速度']
            })

        # 左车道
        if ego_lane < 2:
            left_lane = current_vehicles[current_vehicles['车道'] == ego_lane + 1]
            left_front = left_lane[left_lane['质心X'] > ego_x]
            left_rear = left_lane[left_lane['质心X'] < ego_x]

            if not left_front.empty:
                lf = left_front.loc[left_front['质心X'].idxmin()]
                spacing = lf['质心X'] - ego_x
                surr_info.update({
                    '左前车ID': lf['车辆ID'], '左前车间距': spacing,
                    '左前车纵向速度': lf['纵向速度'], '左前车纵向加速度': lf['纵向加速度']
                })

            if not left_rear.empty:
                lr = left_rear.loc[left_rear['质心X'].idxmax()]
                spacing = ego_x - lr['质心X']
                surr_info.update({
                    '左后车ID': lr['车辆ID'], '左后车间距': spacing,
                    '左后车纵向速度': lr['纵向速度'], '左后车纵向加速度': lr['纵向加速度']
                })

        # 右车道
        if ego_lane > 0:
            right_lane = current_vehicles[current_vehicles['车道'] == ego_lane - 1]
            right_front = right_lane[right_lane['质心X'] > ego_x]
            right_rear = right_lane[right_lane['质心X'] < ego_x]

            if not right_front.empty:
                rf = right_front.loc[right_front['质心X'].idxmin()]
                spacing = rf['质心X'] - ego_x
                surr_info.update({
                    '右前车ID': rf['车辆ID'], '右前车间距': spacing,
                    '右前车纵向速度': rf['纵向速度'], '右前车纵向加速度': rf['纵向加速度']
                })

            if not right_rear.empty:
                rr = right_rear.loc[right_rear['质心X'].idxmax()]
                spacing = ego_x - rr['质心X']
                surr_info.update({
                    '右后车ID': rr['车辆ID'], '右后车间距': spacing,
                    '右后车纵向速度': rr['纵向速度'], '右后车纵向加速度': rr['纵向加速度']
                })

        # 合并本车信息和周围车辆信息
        row_dict = ego.to_dict()
        row_dict.update(surr_info)
        result_rows.append(row_dict)

# 创建结果DataFrame
result_df = pd.DataFrame(result_rows)

# 步骤(5): 排序和格式
column_order = ['车辆ID', '时间戳', '质心X', '质心Y', '车道',
                '纵向速度', '纵向加速度', '横向速度', '横向加速度',
                '车型', '车长', '车宽', '航向角',
                '前车ID', '前车间距', '前车纵向速度', '前车纵向加速度',
                '后车ID', '后车间距', '后车纵向速度', '后车纵向加速度',
                '左前车ID', '左前车间距', '左前车纵向速度', '左前车纵向加速度',
                '左后车ID', '左后车间距', '左后车纵向速度', '左后车纵向加速度',
                '右前车ID', '右前车间距', '右前车纵向速度', '右前车纵向加速度',
                '右后车ID', '右后车间距', '右后车纵向速度', '右后车纵向加速度']

# 只保留存在的列
available_columns = [col for col in column_order if col in result_df.columns]
result_df = result_df[available_columns]

# 保留4位小数
float_cols = ['质心X', '质心Y', '纵向速度', '纵向加速度', '横向速度', '横向加速度',
              '航向角', '前车间距', '前车纵向速度', '前车纵向加速度',
              '后车间距', '后车纵向速度', '后车纵向加速度',
              '左前车间距', '左前车纵向速度', '左前车纵向加速度',
              '左后车间距', '左后车纵向速度', '左后车纵向加速度',
              '右前车间距', '右前车纵向速度', '右前车纵向加速度',
              '右后车间距', '右后车纵向速度', '右后车纵向加速度']

for col in float_cols:
    if col in result_df.columns:
        result_df[col] = result_df[col].round(4)

# 排序
result_df = result_df.sort_values(['车辆ID', '时间戳']).reset_index(drop=True)

# 保存
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"处理完成！已保存到: {output_path}")


# import pandas as pd
# import numpy as np
# from tqdm import tqdm
#
# # 文件路径
# files = [
#     r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845-1处理.csv",
#     r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845-2处理.csv"
# ]
#
# output_files = [
#     r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845-1预处理.csv",
#     r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845-2预处理.csv"
# ]
#
# # 读取并合并两个表的数据
# print("读取数据...")
# dfs = []
# for file in files:
#     df = pd.read_csv(file)
#     dfs.append(df)
#
# # 合并所有数据
# combined_df = pd.concat(dfs, ignore_index=True)
# print(f"合并后总车辆数: {combined_df['车辆ID'].nunique()}")
# print(f"合并后总行数: {len(combined_df)}")
#
#
# # 步骤(2): 确定车道
# def get_lane(y):
#     if 0 < y < 3.5:
#         return 0
#     elif 3.5 < y < 7:
#         return 1
#     elif 7 < y < 10.5:
#         return 2
#     return -1
#
#
# combined_df['车道'] = combined_df['质心Y'].apply(get_lane)
#
#
# # 步骤(4): 计算航向角
# def calculate_heading(group):
#     x = group['质心X'].values
#     y = group['质心Y'].values
#     heading = np.zeros(len(group))
#
#     for i in range(1, len(group)):
#         dx = x[i] - x[i - 1]
#         dy = y[i] - y[i - 1]
#         if dx != 0:
#             heading[i] = np.arctan2(dy, dx)
#         else:
#             heading[i] = np.pi / 2 if dy > 0 else -np.pi / 2
#
#     if len(group) > 1:
#         heading[0] = heading[1]
#
#     group['航向角'] = heading
#     return group
#
#
# combined_df = combined_df.groupby('车辆ID', group_keys=False).apply(calculate_heading)
#
# # 步骤(3): 寻找周围车辆
# # 保存结果列表
# result_rows = []
#
# # 获取所有时间戳
# all_timestamps = sorted(combined_df['时间戳'].unique())
#
# print("正在寻找周围车辆...")
# for ts in tqdm(all_timestamps, desc="处理时间戳"):
#     # 当前时刻的所有车辆
#     current_vehicles = combined_df[combined_df['时间戳'] == ts].copy()
#
#     for _, ego in current_vehicles.iterrows():
#         ego_id = ego['车辆ID']
#         ego_lane = ego['车道']
#         ego_x = ego['质心X']
#
#         # 初始化周围车辆信息
#         surr_info = {
#             '前车ID': -1, '前车间距': np.nan, '前车纵向速度': np.nan, '前车纵向加速度': np.nan,
#             '后车ID': -1, '后车间距': np.nan, '后车纵向速度': np.nan, '后车纵向加速度': np.nan,
#             '左前车ID': -1, '左前车间距': np.nan, '左前车纵向速度': np.nan, '左前车纵向加速度': np.nan,
#             '左后车ID': -1, '左后车间距': np.nan, '左后车纵向速度': np.nan, '左后车纵向加速度': np.nan,
#             '右前车ID': -1, '右前车间距': np.nan, '右前车纵向速度': np.nan, '右前车纵向加速度': np.nan,
#             '右后车ID': -1, '右后车间距': np.nan, '右后车纵向速度': np.nan, '右后车纵向加速度': np.nan
#         }
#
#         # 筛选同车道车辆
#         same_lane = current_vehicles[current_vehicles['车道'] == ego_lane]
#         same_lane = same_lane[same_lane['车辆ID'] != ego_id]
#
#         # 前车
#         front_cars = same_lane[same_lane['质心X'] > ego_x]
#         if not front_cars.empty:
#             front = front_cars.loc[front_cars['质心X'].idxmin()]  # 最近的
#             spacing = front['质心X'] - ego_x
#             surr_info.update({
#                 '前车ID': front['车辆ID'], '前车间距': spacing,
#                 '前车纵向速度': front['纵向速度'], '前车纵向加速度': front['纵向加速度']
#             })
#
#         # 后车
#         rear_cars = same_lane[same_lane['质心X'] < ego_x]
#         if not rear_cars.empty:
#             rear = rear_cars.loc[rear_cars['质心X'].idxmax()]
#             spacing = ego_x - rear['质心X']
#             surr_info.update({
#                 '后车ID': rear['车辆ID'], '后车间距': spacing,
#                 '后车纵向速度': rear['纵向速度'], '后车纵向加速度': rear['纵向加速度']
#             })
#
#         # 左车道
#         if ego_lane < 2:
#             left_lane = current_vehicles[current_vehicles['车道'] == ego_lane + 1]
#             left_front = left_lane[left_lane['质心X'] > ego_x]
#             left_rear = left_lane[left_lane['质心X'] < ego_x]
#
#             if not left_front.empty:
#                 lf = left_front.loc[left_front['质心X'].idxmin()]
#                 spacing = lf['质心X'] - ego_x
#                 surr_info.update({
#                     '左前车ID': lf['车辆ID'], '左前车间距': spacing,
#                     '左前车纵向速度': lf['纵向速度'], '左前车纵向加速度': lf['纵向加速度']
#                 })
#
#             if not left_rear.empty:
#                 lr = left_rear.loc[left_rear['质心X'].idxmax()]
#                 spacing = ego_x - lr['质心X']
#                 surr_info.update({
#                     '左后车ID': lr['车辆ID'], '左后车间距': spacing,
#                     '左后车纵向速度': lr['纵向速度'], '左后车纵向加速度': lr['纵向加速度']
#                 })
#
#         # 右车道
#         if ego_lane > 0:
#             right_lane = current_vehicles[current_vehicles['车道'] == ego_lane - 1]
#             right_front = right_lane[right_lane['质心X'] > ego_x]
#             right_rear = right_lane[right_lane['质心X'] < ego_x]
#
#             if not right_front.empty:
#                 rf = right_front.loc[right_front['质心X'].idxmin()]
#                 spacing = rf['质心X'] - ego_x
#                 surr_info.update({
#                     '右前车ID': rf['车辆ID'], '右前车间距': spacing,
#                     '右前车纵向速度': rf['纵向速度'], '右前车纵向加速度': rf['纵向加速度']
#                 })
#
#             if not right_rear.empty:
#                 rr = right_rear.loc[right_rear['质心X'].idxmax()]
#                 spacing = ego_x - rr['质心X']
#                 surr_info.update({
#                     '右后车ID': rr['车辆ID'], '右后车间距': spacing,
#                     '右后车纵向速度': rr['纵向速度'], '右后车纵向加速度': rr['纵向加速度']
#                 })
#
#         # 合并本车信息和周围车辆信息
#         row_dict = ego.to_dict()
#         row_dict.update(surr_info)
#         result_rows.append(row_dict)
#
# # 创建结果DataFrame
# result_df = pd.DataFrame(result_rows)
#
# # 步骤(5): 排序和格式
# column_order = ['车辆ID', '时间戳', '质心X', '质心Y', '车道',
#                 '纵向速度', '纵向加速度', '横向速度', '横向加速度',
#                 '车型', '车长', '车宽', '航向角',
#                 '前车ID', '前车间距', '前车纵向速度', '前车纵向加速度',
#                 '后车ID', '后车间距', '后车纵向速度', '后车纵向加速度',
#                 '左前车ID', '左前车间距', '左前车纵向速度', '左前车纵向加速度',
#                 '左后车ID', '左后车间距', '左后车纵向速度', '左后车纵向加速度',
#                 '右前车ID', '右前车间距', '右前车纵向速度', '右前车纵向加速度',
#                 '右后车ID', '右后车间距', '右后车纵向速度', '右后车纵向加速度']
#
# # 只保留存在的列
# available_columns = [col for col in column_order if col in result_df.columns]
# result_df = result_df[available_columns]
#
# # 保留4位小数
# float_cols = ['质心X', '质心Y', '纵向速度', '纵向加速度', '横向速度', '横向加速度',
#               '航向角', '前车间距', '前车纵向速度', '前车纵向加速度',
#               '后车间距', '后车纵向速度', '后车纵向加速度',
#               '左前车间距', '左前车纵向速度', '左前车纵向加速度',
#               '左后车间距', '左后车纵向速度', '左后车纵向加速度',
#               '右前车间距', '右前车纵向速度', '右前车纵向加速度',
#               '右后车间距', '右后车纵向速度', '右后车纵向加速度']
#
# for col in float_cols:
#     if col in result_df.columns:
#         result_df[col] = result_df[col].round(4)
#
# # 排序
# result_df = result_df.sort_values(['车辆ID', '时间戳']).reset_index(drop=True)
#
# # 保存到两个表
# print("\n保存结果到两个表...")
#
# # 读取原始表的结构，按原始车辆ID分配
# for i, file in enumerate(files):
#     # 读取原始文件获取车辆ID列表
#     original_df = pd.read_csv(file)
#     original_vehicle_ids = set(original_df['车辆ID'].unique())
#
#     # 筛选属于这个表的车辆数据
#     table_result = result_df[result_df['车辆ID'].isin(original_vehicle_ids)].copy()
#
#     # 保存
#     output_path = output_files[i]
#     table_result.to_csv(output_path, index=False, encoding='utf-8-sig')
#
#     print(f"表{i + 1}: 车辆数={table_result['车辆ID'].nunique()}, 行数={len(table_result)}, 保存到: {output_path}")
#
# print("\n处理完成！")