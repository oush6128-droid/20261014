# import pandas as pd
#
# # 设置显示选项
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
#
# # 文件路径
# preprocess_path = r"D:\桌面\车载组\6-4_徐东航拍数据\第一次\932预处理.csv"
# excel_path = r"D:\桌面\车载组\6-4_徐东航拍数据\第一次\932断面分析统计.xlsx"
# road_length = 1145.8  # 道路总长度
#
# # 读取预处理数据
# print("正在读取预处理数据...")
# preprocess_df = pd.read_csv(preprocess_path)
#
# # 1. 计算空间占有率
# print("\n1. 计算空间占有率...")
#
# # 对齐时间戳
# preprocess_df['时间戳_对齐'] = preprocess_df['时间戳']
#
# # 按对齐后的时间戳和车道分组，计算每个时间戳每个车道上车辆的总长度
# spatial_occupancy_by_time = preprocess_df.groupby(['时间戳_对齐', '车道']).agg({
#     '车长': 'sum'
# }).reset_index()
#
# # 计算空间占有率 (车长总和 / 道路长度)
# spatial_occupancy_by_time['空间占有率'] = spatial_occupancy_by_time['车长'] / road_length * 100
#
# # 将数据透视为时间戳为索引，车道为列
# spatial_pivot = spatial_occupancy_by_time.pivot_table(
#     index='时间戳_对齐',
#     columns='车道',
#     values='空间占有率',
#     fill_value=0
# ).reset_index()
#
# # 重命名列
# spatial_pivot.columns.name = None
# spatial_pivot = spatial_pivot.rename(columns={'时间戳_对齐': '时间戳'})
#
# # 确保有0,1,2三个车道列
# for lane in [0, 1, 2]:
#     if lane not in spatial_pivot.columns:
#         spatial_pivot[lane] = 0
#
# # 重新排序列
# spatial_pivot = spatial_pivot[['时间戳', 0, 1, 2]]
# spatial_pivot = spatial_pivot.rename(columns={0: '0车道', 1: '1车道', 2: '2车道'})
#
# # 2. 计算时间占有率
# print("\n2. 计算时间占有率...")
#
# # 读取断面分析数据
# with pd.ExcelFile(excel_path) as xlsx:
#     section_df = pd.read_excel(xlsx, sheet_name='断面分析')
#
# # 定义断面位置
# sections = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
#
# # 准备存储结果的列表
# time_occupancy_results = []
#
# # 对每个断面计算时间占有率
# for section in sections:
#     time_col = f"{section}米时间"
#     speed_col = f"{section}米速度"
#
#     # 初始化各车道总占用时间
#     lane_occupancy_time = {0: 0, 1: 0, 2: 0}
#
#     # 处理每辆车
#     for idx, row in section_df.iterrows():
#         vehicle_id = row.get('车辆ID', idx)
#         section_time = row[time_col]
#         section_speed = row[speed_col]
#
#         # 检查时间是否为有效值
#         if pd.isna(section_time) or pd.isna(section_speed) or section_speed <= 0:
#             continue
#
#         # 将时间戳对齐到1位小数
#         aligned_time = round(section_time, 1)
#
#         # 查找车辆在此时刻所在的车道
#         vehicle_data = preprocess_df[
#             (preprocess_df['时间戳_对齐'] == aligned_time) &
#             (preprocess_df['车辆ID'] == vehicle_id)
#             ]
#
#         if len(vehicle_data) > 0:
#             lane = int(vehicle_data.iloc[0]['车道'])
#             vehicle_length = vehicle_data.iloc[0]['车长']
#
#             # 计算单辆车的占用时间
#             # 时间占有率 = 车长 / 速度
#             occupancy_time = vehicle_length / section_speed
#
#             # 累加到对应车道
#             lane_occupancy_time[lane] += occupancy_time
#         else:
#             print(f"警告: 车辆{vehicle_id}在时间{aligned_time}的记录未找到")
#
#     # 计算各车道时间占有率百分比
#     total_time = 1688.4  # 总检测时间1800秒
#     lane_occupancy_percent = {}
#     for lane in [0, 1, 2]:
#         lane_occupancy_percent[lane] = (lane_occupancy_time[lane] / total_time) * 100
#
#     # 添加到结果
#     time_occupancy_results.append({
#         '断面': section,
#         '0车道': lane_occupancy_percent[0],
#         '1车道': lane_occupancy_percent[1],
#         '2车道': lane_occupancy_percent[2]
#     })
#
# # 创建时间占有率DataFrame
# time_occupancy_df = pd.DataFrame(time_occupancy_results)
#
# # 3. 保存结果到Excel
# print("\n3. 保存结果到Excel...")
#
# # 使用openpyxl引擎，支持写入
# with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#     # 保存空间占有率
#     spatial_pivot.to_excel(writer, sheet_name='空间占有率', index=False)
#
#     # 保存时间占有率
#     time_occupancy_df.to_excel(writer, sheet_name='时间占有率', index=False)
#
# print(f"结果已保存到: {excel_path}")
# print("已添加工作表: '空间占有率' 和 '时间占有率'")


import pandas as pd
from tqdm import tqdm

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 文件路径
preprocess_files = [
    r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845-1预处理.csv",
    r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845-2预处理.csv"
]
excel_path = r"D:\桌面\车载组\6-4_徐东航拍数据\第三次\1845断面分析统计.xlsx"
road_length = 1139
chunk_size = 100000

# 1. 计算空间占有率
print("正在计算空间占有率...")
all_spatial_data = []

for file_idx, file in enumerate(preprocess_files, 1):
    print(f"处理文件{file_idx}...")

    for chunk in tqdm(pd.read_csv(file, chunksize=chunk_size), desc="分块处理"):
        chunk['时间戳_对齐'] = chunk['时间戳']
        spatial_chunk = chunk.groupby(['时间戳_对齐', '车道']).agg({
            '车长': 'sum'
        }).reset_index()
        spatial_chunk['空间占有率'] = spatial_chunk['车长'] / road_length * 100
        all_spatial_data.append(spatial_chunk)

# 合并所有空间占有率数据
if all_spatial_data:
    spatial_df = pd.concat(all_spatial_data, ignore_index=True)

    # 透视表
    spatial_pivot = spatial_df.pivot_table(
        index='时间戳_对齐',
        columns='车道',
        values='空间占有率',
        fill_value=0
    ).reset_index()

    spatial_pivot.columns.name = None
    spatial_pivot = spatial_pivot.rename(columns={'时间戳_对齐': '时间戳'})

    # 确保有0,1,2三个车道列
    for lane in [0, 1, 2]:
        if lane not in spatial_pivot.columns:
            spatial_pivot[lane] = 0

    spatial_pivot = spatial_pivot[['时间戳', 0, 1, 2]]
    spatial_pivot = spatial_pivot.rename(columns={0: '0车道', 1: '1车道', 2: '2车道'})
else:
    print("没有空间占有率数据")

# 2. 计算时间占有率
print("\n计算时间占有率...")
try:
    with pd.ExcelFile(excel_path) as xlsx:
        section_df = pd.read_excel(xlsx, sheet_name='断面分析')
except:
    print(f"无法读取Excel文件: {excel_path}")
    exit()

sections = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
time_occupancy_results = []

# 创建车辆ID到车长、车道的映射（分块处理）
print("创建车辆映射...")
vehicle_map = {}

for file in preprocess_files:
    for chunk in pd.read_csv(file, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            vehicle_id = row['车辆ID']
            timestamp = round(row['时间戳'], 1)
            lane = int(row['车道'])
            vehicle_length = row['车长']

            key = (vehicle_id, timestamp)
            if key not in vehicle_map:
                vehicle_map[key] = {'lane': lane, 'length': vehicle_length}

print(f"创建了 {len(vehicle_map)} 个车辆时间记录映射")

# 对每个断面计算时间占有率
for section in tqdm(sections, desc="断面计算"):
    time_col = f"{section}米时间"
    speed_col = f"{section}米速度"

    lane_occupancy_time = {0: 0, 1: 0, 2: 0}

    for idx, row in section_df.iterrows():
        vehicle_id = row.get('车辆ID', idx)
        section_time = row[time_col]
        section_speed = row[speed_col]

        if pd.isna(section_time) or pd.isna(section_speed) or section_speed <= 0:
            continue

        aligned_time = round(section_time, 1)
        key = (vehicle_id, aligned_time)

        if key in vehicle_map:
            lane = vehicle_map[key]['lane']
            vehicle_length = vehicle_map[key]['length']
            occupancy_time = vehicle_length / section_speed
            lane_occupancy_time[lane] += occupancy_time

    # 计算百分比
    total_time = 1642.2
    lane_occupancy_percent = {}
    for lane in [0, 1, 2]:
        lane_occupancy_percent[lane] = (lane_occupancy_time[lane] / total_time) * 100

    time_occupancy_results.append({
        '断面': section,
        '0车道': lane_occupancy_percent[0],
        '1车道': lane_occupancy_percent[1],
        '2车道': lane_occupancy_percent[2]
    })

time_occupancy_df = pd.DataFrame(time_occupancy_results)

# 3. 保存结果
print("\n保存结果到Excel...")
try:
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        spatial_pivot.to_excel(writer, sheet_name='空间占有率', index=False)
        time_occupancy_df.to_excel(writer, sheet_name='时间占有率', index=False)
    print(f"结果已保存到: {excel_path}")
except Exception as e:
    print(f"保存失败: {e}")
    # 创建新文件
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        spatial_pivot.to_excel(writer, sheet_name='空间占有率', index=False)
        time_occupancy_df.to_excel(writer, sheet_name='时间占有率', index=False)
    print(f"已创建新文件: {excel_path}")