import pandas as pd

# 读取 CSV 文件（路径和你给的一致）
file_path = r'C:\Users\千里走单骑\Desktop\大论文复现\20260325曾锦鹤大论文与关键数据\第三次轨迹数据\1845-2预处理.csv'

# 读取数据
df = pd.read_csv(file_path)

# 按车辆ID分组，计算横向速度、横向加速度的平均值
result = df.groupby('车辆ID')[['横向速度', '横向加速度']].mean()

# 重命名列名，更清晰
result.columns = ['横向速度平均值', '横向加速度平均值']

# 显示前几行看看
print(result.head())

# 保存到新 CSV（方便论文用）
result.to_csv(r'C:\Users\千里走单骑\Desktop\每辆车横向运动平均值2-2.csv', encoding='utf-8-sig')

print("\n计算完成，已保存到：桌面 / 每辆车横向运动平均值2-2.csv")