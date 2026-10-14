import pandas as pd

csv_path = r"D:\1TEST20260510\指标\MUST CHANGE 1845.csv"
df = pd.read_csv(csv_path)

# 按车辆ID和时间排序，确保顺序正确
df = df.sort_values(['车辆ID', '时间戳'])

# 计算每辆车车道编号的变化（非零即换道）
df['车道变化'] = df.groupby('车辆ID')['车道'].diff().fillna(0)
df['is_lane_change'] = df['车道变化'] != 0

result = df.groupby('车辆ID').agg(
    风格=('驾驶风格', 'first'),
    换道次数=('is_lane_change', 'sum'),
).reset_index()

print(f"总车辆数: {len(result)}")
print(f"\n各风格换道统计:")
print(result.groupby('风格')['换道次数'].describe())
print(f"\n全部结果:")
print(result.to_string())

result.to_csv(r'd:\1TEST20260510\指标\换道次数\1845 MUST CHANGE换道频率_result.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存至: d:\\1TEST20260510\\30-原始932风格换道频率_result.csv")