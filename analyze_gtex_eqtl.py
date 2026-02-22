#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 GTEx 肺组织 eQTL 作为暴露，进行两样本 MR
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_DIR = Path("MR_Data")
OUTCOME_FILE = DATA_DIR / "lung_cancer_outcome.tsv"
EQTL_FILE = DATA_DIR / "GTEx_lung_eqtl_target.tsv"

# 读取结局数据
outcome_df = pd.read_csv(OUTCOME_FILE, sep='\t')
outcome_df = outcome_df[['SNP', 'beta', 'se', 'pval']].rename(
    columns={'beta': 'beta_out', 'se': 'se_out', 'pval': 'pval_out'})

# 读取 eQTL 数据
eqtl_df = pd.read_csv(EQTL_FILE, sep='\t')

results = []
for gene in eqtl_df['gene_name'].unique():
    print(f"\n=== 分析基因: {gene} ===")
    subset = eqtl_df[eqtl_df['gene_name'] == gene].copy()

    # 与结局匹配
    merged = pd.merge(subset, outcome_df, on='SNP', how='inner')
    print(f"  匹配 SNP 数量: {len(merged)}")

    if len(merged) < 3:
        print("  SNP 不足，跳过")
        continue

    # 过滤 beta_exposure 不为 0
    merged = merged[merged['beta'] != 0]
    merged = merged[np.isfinite(merged['beta']) & np.isfinite(merged['beta_out'])]

    if len(merged) < 2:
        print("  有效 SNP 不足")
        continue

    # 计算 Wald 比率
    merged['wald_ratio'] = merged['beta_out'] / merged['beta']
    merged['wald_se'] = np.sqrt(
        (merged['se_out'] ** 2 / merged['beta'] ** 2) +
        (merged['beta_out'] ** 2 * merged['se'] ** 2 / merged['beta'] ** 4)
    )
    merged = merged[np.isfinite(merged['wald_ratio']) & np.isfinite(merged['wald_se'])]

    # IVW 固定效应
    weights = 1 / (merged['wald_se'] ** 2)
    beta_ivw = np.sum(merged['wald_ratio'] * weights) / np.sum(weights)
    se_ivw = np.sqrt(1 / np.sum(weights))
    p_ivw = 2 * (1 - stats.norm.cdf(np.abs(beta_ivw / se_ivw))) if se_ivw > 0 else 1.0

    # F 统计量（暴露）
    f_stat = (merged['beta'] / merged['se']) ** 2
    mean_f = f_stat.mean()

    results.append({
        'gene': gene,
        'n_snps': len(merged),
        'mean_f': mean_f,
        'beta_ivw': beta_ivw,
        'se_ivw': se_ivw,
        'pval_ivw': p_ivw,
        'ci_lower': beta_ivw - 1.96 * se_ivw,
        'ci_upper': beta_ivw + 1.96 * se_ivw
    })

    print(f"  IVW: beta = {beta_ivw:.4f}, se = {se_ivw:.4f}, p = {p_ivw:.4e}, F = {mean_f:.2f}")

# 保存结果
res_df = pd.DataFrame(results)
res_df.to_csv(DATA_DIR / "GTEx_eqtl_MR_results.csv", index=False)
print("\n所有 eQTL MR 分析完成，结果已保存。")