import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免图表中文乱码，若不需要中文可删除）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取 Excel 文件（注意路径中的反斜杠，建议使用原始字符串或双反斜杠）
file_path = r'D:\1TEST20260510\指标\V-T\仅风格.xlsx'
df = pd.read_excel(file_path)   # 若有多张表，可用 sheet_name='Sheet1' 指定

# 2. 提取两个字段，并丢弃缺失值
x = df['密度']        # 自变量
y = df['平均速度']    # 因变量
mask = x.notna() & y.notna()
x = x[mask]
y = y[mask]

# 3. 线性拟合 (一次多项式)
coeffs = np.polyfit(x, y, 1)               # [斜率, 截距]
slope, intercept = coeffs[0], coeffs[1]
line_eq = f'y = {slope:.4f}x + {intercept:.4f}'

# 4. 绘图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6, label='仅风格', color='steelblue', edgecolors='white')

# 生成拟合直线的 x 范围
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, 'r--', label=f'拟合直线：{line_eq}', linewidth=2)

# 5. 图形标注
plt.xlabel('密度')
plt.ylabel('平均速度')
plt.title('密度与速度散点图（原始）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0, 60)      # 横坐标范围
plt.ylim(13, 21)     # 纵坐标范围

# 6. 输出方程并显示图形
print('拟合直线方程：', line_eq)
plt.show()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # 设置中文字体（避免乱码，若不需要可注释掉）
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
# plt.rcParams['axes.unicode_minus'] = False

# # 1. 读取 Excel 文件
# file_path = r'D:\1TEST20260510\指标\V-T\MUST CHANGE.xlsx'
# df = pd.read_excel(file_path)          # 如有多个工作表，请添加 sheet_name='Sheet1'

# # 2. 提取所需列（密度为 x，平均速度为 y），并剔除缺失值
# x = df['密度']
# y = df['平均速度']
# mask = x.notna() & y.notna()
# x = x[mask]
# y = y[mask]

# # 3. 指定的直线参数
# slope = -0.0560
# intercept = 19.8142
# line_eq = f'y = {slope:.4f}x + {intercept:.4f}'

# # 4. 绘图：散点 + 指定直线
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, alpha=0.6, label='原始数据', color='steelblue', edgecolors='white')

# # 生成直线的 x 范围（覆盖数据范围）
# x_line = np.linspace(x.min(), x.max(), 100)
# y_line = slope * x_line + intercept
# plt.plot(x_line, y_line, 'r--', label=f'指定直线：{line_eq}', linewidth=2)

# # 5. 图形标注
# plt.xlabel('密度')
# plt.ylabel('平均速度')
# plt.title('密度与速度散点图（加入驾驶目的）')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.xlim(0, 60)      # 横坐标范围
# plt.ylim(13, 21)     # 纵坐标范围
# # 6. 显示结果
# print('已绘制的直线方程：', line_eq)
# plt.show()