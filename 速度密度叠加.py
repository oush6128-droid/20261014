import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 三条直线：斜率, 截距, 颜色, 标签
lines = [
    (-0.0507, 19.6622, 'red',   '原始'),
    (-0.0643, 19.4742, 'blue',  '仅风格'),
    (-0.0560, 19.8142, 'green', '含目的')
]

# 密度 x 范围 0~200
x = np.linspace(0, 200, 300)

plt.figure(figsize=(9, 6))

for m, b, color, label in lines:
    y = m * x + b          # 速度
    plt.plot(x, y, color=color, linewidth=2, label=label)

plt.xlabel('密度')
plt.ylabel('速度')
plt.title('速度 - 密度关系')
plt.xlim(0, 200)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()