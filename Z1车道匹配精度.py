import pandas as pd

# 读取CSV，处理BOM
df = pd.read_csv(r"D:\1TEST20260510\指标\MUST CHANGE 932.csv", encoding='utf-8-sig')

print(f"总行数: {len(df)}")
print(f"车辆数: {df['车辆ID'].nunique()}")
print()

# 按车辆ID分组，取每辆车的第一帧和最后一帧
result = df.groupby('车辆ID').agg(
    初始车道=('车道', 'first'),
    最终车道=('车道', 'last'),
    初始帧=('帧序号', 'first'),
    最终帧=('帧序号', 'last'),
    最终X=('质心X', 'last'),
    驾驶风格=('驾驶风格', 'first'),
).reset_index()

# 仅保留走完全程的车辆（最终X > 2900）
result = result[result['最终X'] > 2900]
print(f"走完全程的车辆: {len(result)}")

# 目的地车道 = 初始车道（0→0, 2→2, 1→1），不用CSV里的假字段
def get_dest_lane(init_lane):
    if init_lane == 0:
        return 2
    elif init_lane == 2:
        return 0
    return 1

result['目的地车道'] = result['初始车道'].apply(get_dest_lane)

# 判断是否到达目的地车道
result['是否到达目的地'] = result['最终车道'] == result['目的地车道']

print("每辆车的初始车道和最终车道：")
print("=" * 80)
for _, row in result.iterrows():
    flag = "✓" if row['是否到达目的地'] else "✗"
    print(f"车辆ID={int(row['车辆ID']):3d}  |  初始车道={row['初始车道']}  →  最终车道={row['最终车道']}  |  "
          f"目的地车道={row['目的地车道']}  {flag}  |  风格={row['驾驶风格']}  |  帧范围={int(row['初始帧'])}~{int(row['最终帧'])}")

print()
print("=" * 80)
arrived = result['是否到达目的地'].sum()
total = len(result)
print(f"到达目的地车道的车辆: {arrived}/{total} ({arrived/total*100:.1f}%)")

# 统计换道情况
result['是否换道'] = result['初始车道'] != result['最终车道']
changed = result['是否换道'].sum()
print(f"发生换道的车辆: {changed}/{total} ({changed/total*100:.1f}%)")

# 详细统计：各车道的初始-最终分布
print()
print("各车辆车道变化详情（仅显示发生了变化的）：")
changed_vehicles = result[result['是否换道']]
if len(changed_vehicles) > 0:
    for _, row in changed_vehicles.iterrows():
        print(f"  车辆{int(row['车辆ID']):3d}: 车道 {row['初始车道']} → {row['最终车道']} (目的地={row['目的地车道']})")
else:
    print("  没有车辆发生车道变化")

print()
print("=" * 80)
print("三个场景统计（0→0 / 1→1 / 2→2）：")
print(f"{'初始→目的':<12} {'车辆数':<8} {'到达数':<8} {'精度':<10}")
print("-" * 50)
for init_lane, dest_lane in [(0, 0), (1, 1), (2, 2)]:
    subset = result[(result['初始车道'] == init_lane) & (result['目的地车道'] == dest_lane)]
    n = len(subset)
    n_arrived = subset['是否到达目的地'].sum()
    acc = n_arrived / n * 100 if n > 0 else 0
    print(f"{init_lane}→{dest_lane}       {n:<8} {n_arrived:<8} {acc:<9.1f}%")
