import matplotlib.pyplot as plt
import numpy as np

# 层数（1 到 8）
layers = np.arange(1, 9)

# 假设每个层数对应的 AUC（你需要替换成自己的真实数据）
auc_values = [0.9321, 0.9358, 0.9380, 0.9429, 0.9412, 0.9389, 0.9365, 0.9340]

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(layers, auc_values, marker='o', linestyle='-', linewidth=2.5, color='#4C72B0', markersize=8)

# 添加每个点的顶部数值
for x, y in zip(layers, auc_values):
    plt.text(x, y + 0.0006, f'{y:.4f}', ha='center', va='bottom', fontsize=14)

# 设置横轴为层数
plt.xticks(layers, [str(i) for i in layers], fontsize=15)
plt.xlabel('Number of Layers', fontsize=18)

# 设置纵轴为 AUC
plt.ylim(0.924, 0.950)
plt.yticks(np.arange(0.924, 0.952, 0.002), fontsize=12)
plt.ylabel('AUC', fontsize=18)

# 网格与边框美化
plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.8, alpha=0.6)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.show()
