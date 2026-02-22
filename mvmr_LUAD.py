#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多变量 MR：LUAD 中 IL18 + IL18R1 对肺癌的独立效应
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from statsmodels.api import add_constant, WLS

WORK_DIR = Path("C:/Users/wswan/PycharmProjects/PythonProject2")
OUTCOME_FILE = WORK_DIR / "MR_Data/lung_cancer_outcome.tsv"
TCGA_FILE = WORK_DIR / "TCGA_eQTL_MR/LUAD_cis_target_eqtl.tsv"
OUTPUT_DIR = WORK_DIR / "TCGA_MR_Results"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_exposure(file_path, gene):
    df = pd.read_csv(file_path, sep='\t')
    df['gene'] = df['gene'].str.split('|').str[0]
    df = df[df['gene'] == gene].copy()
    df = df.rename(columns={'beta': 'beta_exp', 'se': 'se_exp'})
    return df[['SNP', 'beta_exp', 'se_exp']]


def multivariate_mv_ivw(exposure_dfs, exposure_names, outcome_df):
    # 合并所有 SNP
    all_snps = set()
    for df in exposure_dfs:
        all_snps.update(df['SNP'])
    all_snps = list(all_snps)

    X_list = []
    for snp in all_snps:
        row = {'SNP': snp}
        for name, df in zip(exposure_names, exposure_dfs):
            sub = df[df['SNP'] == snp]
            if len(sub) == 1:
                row[f'beta_{name}'] = sub.iloc[0]['beta_exp']
                row[f'se_{name}'] = sub.iloc[0]['se_exp']
            else:
                row[f'beta_{name}'] = 0
                row[f'se_{name}'] = 1e6  # 权重极小
        X_list.append(row)
    X_df = pd.DataFrame(X_list)

    merged = pd.merge(X_df, outcome_df, on='SNP', how='inner')
    if len(merged) < 3:
        print("SNP 数量不足")
        return None

    beta_cols = [f'beta_{name}' for name in exposure_names]
    X = merged[beta_cols].values
    y = merged['beta_out'].values
    weights = 1 / (merged['se_out'] ** 2)

    X_const = add_constant(X)
    model = WLS(y, X_const, weights=weights)
    results = model.fit()

    res = []
    for i, name in enumerate(['intercept'] + exposure_names):
        beta = results.params[i]
        se = results.bse[i]
        p = results.pvalues[i]
        ci_lower = beta - 1.96 * se
        ci_upper = beta + 1.96 * se
        res.append({
            'exposure': name,
            'beta': beta,
            'se': se,
            'pval': p,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_snps': len(merged)
        })
    return pd.DataFrame(res)


def main():
    if not TCGA_FILE.exists():
        print("TCGA 文件不存在")
        return
    if not OUTCOME_FILE.exists():
        print("结局文件不存在")
        return

    outcome = pd.read_csv(OUTCOME_FILE, sep='\t')
    outcome = outcome[['SNP', 'beta', 'se']].rename(
        columns={'beta': 'beta_out', 'se': 'se_out'})

    df_il18 = load_exposure(TCGA_FILE, 'IL18')
    df_il18r1 = load_exposure(TCGA_FILE, 'IL18R1')

    print(f"IL18 SNP 数: {len(df_il18)}")
    print(f"IL18R1 SNP 数: {len(df_il18r1)}")

    res = multivariate_mv_ivw(
        [df_il18, df_il18r1],
        ['IL18', 'IL18R1'],
        outcome
    )
    if res is not None:
        print("\n多变量 MR 结果（LUAD）：")
        print(res.to_string(index=False))
        res.to_csv(OUTPUT_DIR / "mvmr_LUAD_IL18_IL18R1.csv", index=False)
        print(f"\n结果已保存至: {OUTPUT_DIR / 'mvmr_LUAD_IL18_IL18R1.csv'}")


if __name__ == "__main__":
    main()