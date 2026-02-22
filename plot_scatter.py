"""
免疫基因共定位与SMR结果可视化（修正版2）
输入文件：immune_genes_coloc_smr_results.csv
输出图表：coloc_bar.png, smr_forest.png, smr_volcano.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置绘图风格
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 12

# 读取结果文件
df = pd.read_csv('immune_genes_coloc_smr_results.csv')
print("原始数据：")
print(df)

# ---------- 1. PP.H4 条形图（取 -log10 转换） ----------
df_coloc = df[df['PP.H4'].notna()].copy()
df_coloc['neg_log10_PP4'] = -np.log10(df_coloc['PP.H4'].clip(lower=1e-300))

plt.figure(figsize=(10, 6))
bars = plt.bar(df_coloc['gene'], df_coloc['neg_log10_PP4'], color='steelblue')
plt.xlabel('Gene')
plt.ylabel('-log10(PP.H4)')
plt.title('Colocalization Posterior Probability (PP.H4) for Immune Genes')
for bar, pp4 in zip(bars, df_coloc['PP.H4']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{pp4:.2e}', ha='center', va='bottom', fontsize=9, rotation=45)
plt.axhline(y=-np.log10(0.8), color='red', linestyle='--', label='PP.H4 = 0.8 threshold')
plt.legend()
plt.tight_layout()
plt.savefig('coloc_bar.png', dpi=150)
plt.show()

# ---------- 2. SMR 效应森林图 ----------
df_smr = df[df['SMR_beta'].notna() & df['SMR_pval'].notna()].copy()
if len(df_smr) == 0:
    print("没有可用的 SMR 数据")
else:
    # 使用更稳定的 isf 方法计算 z 分数，避免浮点问题
    df_smr['z'] = stats.norm.isf(df_smr['SMR_pval'] / 2)
    df_smr['SMR_se_approx'] = np.abs(df_smr['SMR_beta']) / df_smr['z']
    # 确保标准误为正（极小时替换为极小正数）
    df_smr['SMR_se_approx'] = df_smr['SMR_se_approx'].clip(lower=1e-10)
    # 计算 95% 置信区间
    df_smr['ci_lower'] = df_smr['SMR_beta'] - 1.96 * df_smr['SMR_se_approx']
    df_smr['ci_upper'] = df_smr['SMR_beta'] + 1.96 * df_smr['SMR_se_approx']

    plt.figure(figsize=(8, 6))
    df_smr = df_smr.sort_values('SMR_beta', ascending=False)
    y_pos = np.arange(len(df_smr))
    # 绘制置信区间（xerr 必须为正）
    plt.errorbar(df_smr['SMR_beta'], y_pos,
                 xerr=1.96 * df_smr['SMR_se_approx'], fmt='o', color='darkgreen',
                 capsize=5, markersize=8)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.yticks(y_pos, df_smr['gene'])
    plt.xlabel('SMR effect size (beta) with 95% CI (approx)')
    plt.ylabel('Gene')
    plt.title('SMR Causal Effect of Gene Expression on Lung Cancer')
    plt.tight_layout()
    plt.savefig('smr_forest.png', dpi=150)
    plt.show()

# ---------- 3. SMR 火山图 ----------
if len(df_smr) > 0:
    plt.figure(figsize=(8, 6))
    df_smr['neg_log10_pval'] = -np.log10(df_smr['SMR_pval'].clip(lower=1e-300))
    colors = ['red' if p < 0.05 else 'blue' for p in df_smr['SMR_pval']]
    plt.scatter(df_smr['SMR_beta'], df_smr['neg_log10_pval'], c=colors, s=100, alpha=0.7)
    plt.xlabel('SMR effect size (beta)')
    plt.ylabel('-log10(SMR p-value)')
    plt.title('SMR Volcano Plot: Gene Expression vs Lung Cancer')
    plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--', label='p = 0.05')
    plt.axvline(x=0, color='gray', linestyle='--')
    for _, row in df_smr.iterrows():
        plt.text(row['SMR_beta'], row['neg_log10_pval']+0.05, row['gene'],
                 fontsize=9, ha='center')
    plt.legend()
    plt.tight_layout()
    plt.savefig('smr_volcano.png', dpi=150)
    plt.show()

print("所有图表已保存。")