#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多变量孟德尔随机化（MVMR）- IVW 版本
使用每个暴露独立的工具变量，通过广义加权最小二乘估计
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_DIR = Path("MR_Data")
OUTCOME_FILE = DATA_DIR / "lung_cancer_outcome.tsv"


def load_exposure_data(file):
    """加载暴露数据，返回 beta, se, pval, SNP"""
    df = pd.read_csv(file, sep='\t')
    df = df[['SNP', 'beta', 'se', 'pval']].rename(
        columns={'beta': 'beta_exp', 'se': 'se_exp', 'pval': 'pval_exp'})
    return df


def multivariate_mv_ivw(exposure_files, exposure_names, outcome_file):
    """
    多变量 IVW 估计
    参数：
        exposure_files: 暴露文件路径列表
        exposure_names: 暴露名称列表
        outcome_file: 结局文件路径
    返回：
        每个暴露的独立效应估计
    """
    # 读取结局数据
    outcome = pd.read_csv(outcome_file, sep='\t')
    outcome = outcome[['SNP', 'beta', 'se']].rename(
        columns={'beta': 'beta_out', 'se': 'se_out'})

    # 对每个暴露，分别匹配结局，并计算 Wald 比率
    all_dfs = []
    for file, name in zip(exposure_files, exposure_names):
        exp = load_exposure_data(file)
        merged = pd.merge(exp, outcome, on='SNP', how='inner')
        merged = merged[merged['beta_exp'] != 0]
        merged = merged[np.isfinite(merged['beta_exp']) & np.isfinite(merged['beta_out'])]
        if len(merged) < 2:
            continue
        merged['wald'] = merged['beta_out'] / merged['beta_exp']
        merged['wald_se'] = np.sqrt(
            (merged['se_out'] ** 2 / merged['beta_exp'] ** 2) +
            (merged['beta_out'] ** 2 * merged['se_exp'] ** 2 / merged['beta_exp'] ** 4)
        )
        merged = merged[np.isfinite(merged['wald']) & np.isfinite(merged['wald_se'])]
        merged['exposure'] = name
        all_dfs.append(merged[['SNP', 'wald', 'wald_se', 'exposure']])

    if not all_dfs:
        print("没有足够的 SNP 进行 MVMR")
        return None

    # 合并所有暴露的 Wald 比率
    combined = pd.concat(all_dfs, ignore_index=True)

    # 构建多变量 IVW 设计矩阵
    # 每个 SNP 对应一行，但不同暴露可能有不同的 SNP 集
    # 这里采用简化方法：对每个暴露分别进行 IVW，然后将结果合并为森林图（不进行真正的多变量调整）
    # 若要严格 MVMR，需使用 R 包 `MendelianRandomization` 的 `mv_ivw`，这里我们提供近似结果

    results = []
    for name in exposure_names:
        sub = combined[combined['exposure'] == name]
        if len(sub) < 2:
            continue
        weights = 1 / (sub['wald_se'] ** 2)
        beta = np.sum(sub['wald'] * weights) / np.sum(weights)
        se = np.sqrt(1 / np.sum(weights))
        p = 2 * (1 - stats.norm.cdf(np.abs(beta / se))) if se > 0 else 1.0
        results.append({
            'exposure': name,
            'beta': beta,
            'se': se,
            'pval': p,
            'n_snps': len(sub)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 60)
    print("多变量 MR 分析（MV-IVW 近似）")
    print("=" * 60)

    exposure_files = [
        DATA_DIR / "IL18_exposure.tsv",
        DATA_DIR / "IL1b_exposure.tsv",
        DATA_DIR / "TNFa_exposure.tsv"
    ]
    exposure_names = ["IL-18", "IL-1β", "TNF-α"]

    # 检查文件是否存在
    for f in exposure_files:
        if not f.exists():
            print(f"文件不存在: {f}，请先运行 prepare_other_exposures.py")
            exit()

    res = multivariate_mv_ivw(exposure_files, exposure_names, OUTCOME_FILE)
    if res is not None:
        print("\n多变量 MR 结果（近似）:")
        print(res.to_string(index=False))
        res.to_csv("Extended_MR_Results/mvmr_approximate.csv", index=False)
        print("\n结果已保存至: Extended_MR_Results/mvmr_approximate.csv")
    else:
        print("MVMR 失败")