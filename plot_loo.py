#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
留一法敏感性分析森林图（组合图：LUAD + LUSC）
生成一个包含两个子图的图形，上图为LUAD，下图为LUSC。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def load_and_merge(exposure_file, outcome_file, sep='\t'):
    """
    加载暴露和结局数据，合并并返回DataFrame
    """
    if not os.path.exists(exposure_file):
        raise FileNotFoundError(f"暴露文件不存在: {exposure_file}")
    if not os.path.exists(outcome_file):
        raise FileNotFoundError(f"结局文件不存在: {outcome_file}")

    exp = pd.read_csv(exposure_file, sep=sep)
    out = pd.read_csv(outcome_file, sep=sep)

    # 重命名列（根据实际列名调整）
    exp = exp[['SNP', 'beta', 'se']].rename(columns={'beta': 'beta_exp', 'se': 'se_exp'})
    out = out[['SNP', 'beta', 'se']].rename(columns={'beta': 'beta_out', 'se': 'se_out'})

    merged = pd.merge(exp, out, on='SNP', how='inner')
    merged = merged[(merged['beta_exp'] != 0) & np.isfinite(merged['beta_exp'])]
    print(f"{exposure_file}: 合并后SNP数量 {len(merged)}")
    return merged


def ivw_estimate(df):
    """
    IVW固定效应估计
    """
    beta_exp = df['beta_exp'].values
    se_exp = df['se_exp'].values
    beta_out = df['beta_out'].values
    se_out = df['se_out'].values

    wald = beta_out / beta_exp
    wald_se = np.sqrt((se_out ** 2 / beta_exp ** 2) + (beta_out ** 2 * se_exp ** 2 / beta_exp ** 4))

    valid = np.isfinite(wald) & np.isfinite(wald_se) & (wald_se > 0)
    if np.sum(valid) < 2:
        return None
    wald = wald[valid]
    wald_se = wald_se[valid]

    weights = 1 / (wald_se ** 2)
    beta = np.sum(wald * weights) / np.sum(weights)
    se = np.sqrt(1 / np.sum(weights))
    pval = 2 * (1 - stats.norm.cdf(np.abs(beta / se)))
    return {'beta': beta, 'se': se, 'pval': pval, 'n': len(wald)}


def leaveoneout_analysis(merged_df):
    """
    对合并数据框进行留一法分析
    """
    results = []
    total = ivw_estimate(merged_df)
    if total is None:
        raise ValueError("无法计算总体IVW")

    for i in range(len(merged_df)):
        df_loo = merged_df.drop(index=i).reset_index(drop=True)
        res = ivw_estimate(df_loo)
        if res is not None:
            results.append({
                'SNP': merged_df.loc[i, 'SNP'],
                'beta': res['beta'],
                'se': res['se'],
                'pval': res['pval'],
                'n_snps': res['n']
            })
    loo_df = pd.DataFrame(results)
    # 添加总体行
    total_row = pd.DataFrame([{
        'SNP': 'All SNPs',
        'beta': total['beta'],
        'se': total['se'],
        'pval': total['pval'],
        'n_snps': total['n']
    }])
    loo_df = pd.concat([loo_df, total_row], ignore_index=True)
    return loo_df


def plot_leaveoneout_subplots(loo_dict, title_prefix="", figsize=(8, 12), save_path=None):
    """
    loo_dict: 字典，键为子图标签（如 'LUAD', 'LUSC'），值为对应的留一法DataFrame
    绘制上下两个子图
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for ax, (label, loo_df) in zip(axes, loo_dict.items()):
        # 排序
        loo_df_sorted = loo_df.sort_values('beta', ascending=True).reset_index(drop=True)
        snps = loo_df_sorted['SNP'].tolist()
        betas = loo_df_sorted['beta'].values
        errors = loo_df_sorted['se'].values

        # 找出All SNPs位置
        try:
            all_idx = snps.index('All SNPs')
        except ValueError:
            all_idx = -1

        y_pos = np.arange(len(snps))
        ax.errorbar(betas, y_pos, xerr=1.96 * errors, fmt='o', color='steelblue',
                    ecolor='gray', capsize=2, markersize=6)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

        if all_idx >= 0:
            total_beta = betas[all_idx]
            ax.axvline(x=total_beta, color='red', linestyle='-', linewidth=1.5, alpha=0.8)
            ax.plot(total_beta, all_idx, 'D', color='red', markersize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(snps, fontsize=8)
        ax.invert_yaxis()
        ax.set_ylabel('SNP removed', fontsize=10)
        ax.set_title(f"({label})", loc='left', fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

    axes[-1].set_xlabel('MR effect size (β) for lung cancer', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig, axes


if __name__ == "__main__":
    # 设置文件路径（请修改为您的实际路径）
    base_dir = r"C:/Users/wangning/PycharmProjects/pythonProject1/"
    luad_file = os.path.join(base_dir, "LUAD_IL18_coloc_input.tsv")
    lusc_file = os.path.join(base_dir, "LUSC_IL18_coloc_input.tsv")
    outcome_file = os.path.join(base_dir, "lung_cancer_outcome.tsv")  # 请替换为实际结局文件

    # 检查文件存在
    for f in [luad_file, lusc_file, outcome_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"文件不存在: {f}")

    # 加载数据
    merged_luad = load_and_merge(luad_file, outcome_file)
    merged_lusc = load_and_merge(lusc_file, outcome_file)

    # 留一法分析
    loo_luad = leaveoneout_analysis(merged_luad)
    loo_lusc = leaveoneout_analysis(merged_lusc)

    # 保存结果（可选）
    loo_luad.to_csv(os.path.join(base_dir, "LUAD_IL18_leaveoneout.csv"), index=False)
    loo_lusc.to_csv(os.path.join(base_dir, "LUSC_IL18_leaveoneout.csv"), index=False)

    # 绘制组合图
    loo_dict = {'A) LUAD': loo_luad, 'B) LUSC': loo_lusc}
    plot_leaveoneout_subplots(loo_dict, save_path=os.path.join(base_dir, "Supplementary_Figure1_LOO_combined.png"))