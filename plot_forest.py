import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 读取结果
df = pd.read_csv('mr_results.csv')
# 仅保留有工具变量的暴露
df = df[df['n_iv'] > 0].copy()

# 根据 p 值计算近似标准误（用于可视化）
def approx_se_from_pval(beta, pval):
    # 双尾检验：z = beta/se, p = 2*(1 - Phi(|z|))
    z = stats.norm.isf(pval/2)
    se = np.abs(beta) / z
    return se

df['IVW_se_approx'] = df.apply(lambda row: approx_se_from_pval(row['IVW_beta'], row['IVW_pval']), axis=1)
# 计算 95% 置信区间
df['ci_lower'] = df['IVW_beta'] - 1.96 * df['IVW_se_approx']
df['ci_upper'] = df['IVW_beta'] + 1.96 * df['IVW_se_approx']

# 绘制森林图
plt.figure(figsize=(8, 6))
y_pos = np.arange(len(df))
plt.errorbar(df['IVW_beta'], y_pos,
             xerr=1.96*df['IVW_se_approx'],
             fmt='o', capsize=5, markersize=8, color='darkblue')
plt.axvline(0, color='red', linestyle='--', linewidth=1)
plt.yticks(y_pos, df['exposure'])
plt.xlabel('IVW causal effect (β) with 95% CI')
plt.title('Immune cell traits on lung cancer')
plt.tight_layout()
plt.savefig('mr_forest.png', dpi=150)
plt.show()
print("森林图已保存为 mr_forest.png")