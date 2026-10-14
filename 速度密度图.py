import pandas as pd
import os

df = pd.read_csv(r'D:\1TEST20260510\指标\MUST CHANGE 932.csv')

# 时间窗口：0-10, 10-20, ..., 490-500，共50个窗口
df['时间窗口'] = (df['时间戳'] // 10).astype(int) * 10

results = []
for w in sorted(df['时间窗口'].unique()):
    chunk = df[df['时间窗口'] == w]
    avg_speed = chunk['纵向速度'].mean()
    # 密度 = 车辆数 / 路段长度(km)，路段固定2000m
    vehicle_count = chunk['车辆ID'].nunique()
    density = vehicle_count / 2  # veh/km
    results.append({'时间窗口': w, '平均速度': avg_speed, '密度': density})

result_df = pd.DataFrame(results)

os.makedirs(r'D:\1TEST20260510\指标\V-T', exist_ok=True)
result_df.to_csv(r'D:\1TEST20260510\指标\V-T\MUST CHANGE 932 VT.csv', index=False, encoding='utf-8-sig')

print(f'共 {len(result_df)} 行')
print(result_df.head(10))
