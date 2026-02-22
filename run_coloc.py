import pandas as pd
import matplotlib.pyplot as plt

# 读取共定位结果
coloc = pd.read_csv('coloc_results.csv')

# 移除 PP4 为 NaN 的基因（即未分析成功的）
coloc = coloc.dropna(subset=['PP4']).copy()

# 按 PP4 降序排列
coloc = coloc.sort_values('PP4', ascending=False)

# 绘图
plt.figure(figsize=(8, 5))
bars = plt.bar(coloc['gene'], coloc['PP4'], color='steelblue')
plt.axhline(y=0.8, color='red', linestyle='--', label='PP4 = 0.8')
plt.xlabel('Gene')
plt.ylabel('Posterior probability (PP4)')
plt.title('Colocalization of immune genes with lung cancer')
plt.legend()
# 在柱子上方标注数值（科学计数法，避免过小数值显示为0）
for bar, pp4 in zip(bars, coloc['PP4']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{pp4:.2e}', ha='center', va='bottom', fontsize=8, rotation=45)
plt.tight_layout()
plt.savefig('coloc_bar.png', dpi=150)
plt.show()
print("共定位条形图已保存为 coloc_bar.png")