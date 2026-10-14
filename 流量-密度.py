import numpy as np
import matplotlib.pyplot as plt

# 中文字体（可选，若乱码可注释掉）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 三条直线的参数：斜率, 截距, 颜色, 方程文本
lines = [
    (-0.0507, 19.6622, 'red',   '原始'),
    (-0.0643, 19.4742, 'blue',  '仅风格'),
    (-0.0560, 19.8142, 'green', '含风格')
]

# 生成横坐标：密度 x，范围 0~200
x = np.linspace(0, 100, 500)

plt.figure(figsize=(9, 6))

for m, b, color, label in lines:
    q = x * (m * x + b)      # 流量 = x * y
    plt.plot(x, q, color=color, linewidth=2, label=label)

# 图形标注
plt.xlabel('密度')
plt.ylabel('流量')
plt.title('密度 - 流量关系曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# 固定横坐标范围到 200（纵坐标可自动缩放，或按需调整）
plt.xlim(0, 100)
# plt.ylim(0, 1000)   # 如需手动固定纵坐标范围可取消注释

plt.tight_layout()
plt.show()