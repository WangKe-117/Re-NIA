import matplotlib.pyplot as plt
import numpy as np

# 模型 & 数据
models = ['Re-NIA', 'w/o-RSAM', 'w/o-BiCAN', 'w/o-Relearning']
auc_values = [0.8845, 0.8646, 0.8606, 0.8789]
colors = ['#4C72B0', '#55A868', '#F28E2B', '#C44E52']

# 设置精确间距和柱宽
gap = 1.0       # 每段空隙长度
bar_width = 1.5 # 每根柱子的宽度
n_bars = len(models)

# 计算柱子中心的位置（gap + 0.5*bar_width + i*(bar_width + gap)）
x = [gap + bar_width/2 + i*(bar_width + gap) for i in range(n_bars)]

# 计算总图宽度（边界控制）
total_width = gap + n_bars * (bar_width + gap)
xlim_min = 0
xlim_max = total_width

# 绘图
plt.figure(figsize=(8, 6))
bars = plt.bar(x, auc_values, width=bar_width, color=colors)

# 设置 X 轴
plt.xticks(x, models, fontsize=15)
plt.xlim(xlim_min, xlim_max)

# 设置 Y 轴
plt.ylim(0.856, 0.888)
plt.yticks(np.arange(0.856, 0.8882, 0.002), fontsize=12)

# 添加顶部数值
for xi, auc in zip(x, auc_values):
    offset = 0.0006
    plt.text(xi, auc + offset, f'{auc:.4f}', ha='center', va='bottom', fontsize=14)

for spine in plt.gca().spines.values():
    spine.set_linewidth(1.2)  # 默认是 1，可以改成 1.5 或 2

plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.8, alpha=0.6)

# 坐标轴标题
plt.xlabel('', fontsize=18)
plt.ylabel('Recall', fontsize=18)

# 布局优化
plt.tight_layout()
plt.show()
