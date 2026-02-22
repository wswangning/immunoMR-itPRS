#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCGA eQTL MR 分析（完整修复版）
- 支持多 SNP IVW（指定列名）
- 自动输出单 SNP Wald ratio
- 详细调试信息
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ========== 配置 ==========
DATA_DIR = Path("MR_Data")
OUTCOME_FILE = DATA_DIR / "lung_cancer_outcome.tsv"
TCGA_DIR = Path("TCGA_eQTL_MR")
RESULTS_DIR = Path("TCGA_MR_Results")
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================

def mr_ivw(df, beta_exp_col, se_exp_col, beta_out_col, se_out_col):
    """
    逆方差加权法（固定效应）- 必须指定列名
    """
    # 提取列
    beta_exp = df[beta_exp_col].values
    se_exp = df[se_exp_col].values
    beta_out = df[beta_out_col].values
    se_out = df[se_out_col].values

    # 移除无效值（beta_exp 不能为 0，不能为 NaN）
    valid = (beta_exp != 0) & np.isfinite(beta_exp) & np.isfinite(beta_out) & (se_exp > 0) & (se_out > 0)
    df_valid = df[valid].copy()
    n_valid = len(df_valid)
    print(f"    IVW 有效 SNP 数: {n_valid}")

    if n_valid < 2:
        print(f"    IVW 失败: 有效 SNP 不足 2 个")
        return None

    beta_exp_v = df_valid[beta_exp_col].values
    se_exp_v = df_valid[se_exp_col].values
    beta_out_v = df_valid[beta_out_col].values
    se_out_v = df_valid[se_out_col].values

    # Wald ratio 及标准误
    wald_ratio = beta_out_v / beta_exp_v
    wald_se = np.sqrt(
        (se_out_v ** 2 / beta_exp_v ** 2) +
        (beta_out_v ** 2 * se_exp_v ** 2 / beta_exp_v ** 4)
    )
    # 移除可能出现的 NaN/Inf
    finite = np.isfinite(wald_ratio) & np.isfinite(wald_se) & (wald_se > 0)
    wald_ratio = wald_ratio[finite]
    wald_se = wald_se[finite]

    if len(wald_ratio) < 2:
        print(f"    IVW 失败: 有效 Wald ratio 不足 2 个")
        return None

    # IVW 固定效应
    weights = 1 / (wald_se ** 2)
    beta_ivw = np.sum(wald_ratio * weights) / np.sum(weights)
    se_ivw = np.sqrt(1 / np.sum(weights))
    p_ivw = 2 * (1 - stats.norm.cdf(np.abs(beta_ivw / se_ivw)))

    ci_lower = beta_ivw - 1.96 * se_ivw
    ci_upper = beta_ivw + 1.96 * se_ivw

    return {
        'beta': beta_ivw,
        'se': se_ivw,
        'pval': p_ivw,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_snps': len(wald_ratio)
    }


def wald_ratio_single(df, beta_exp_col, se_exp_col, beta_out_col, se_out_col):
    """单个 SNP 的 Wald ratio 估计"""
    beta_exp = df[beta_exp_col].iloc[0]
    se_exp = df[se_exp_col].iloc[0]
    beta_out = df[beta_out_col].iloc[0]
    se_out = df[se_out_col].iloc[0]

    if beta_exp == 0 or not np.isfinite(beta_exp) or not np.isfinite(beta_out):
        return None

    beta = beta_out / beta_exp
    se = np.sqrt(
        (se_out ** 2 / beta_exp ** 2) +
        (beta_out ** 2 * se_exp ** 2 / beta_exp ** 4)
    )
    p = 2 * (1 - stats.norm.cdf(np.abs(beta / se))) if se > 0 else 1.0
    ci_lower = beta - 1.96 * se
    ci_upper = beta + 1.96 * se

    return {
        'beta': beta,
        'se': se,
        'pval': p,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_snps': 1
    }


def analyze_exposure(exposure_file, outcome_file, cancer_type, eqtl_type):
    """分析单个暴露文件"""
    print(f"\n{'=' * 60}")
    print(f"分析: {cancer_type} {eqtl_type} eQTL")
    print(f"{'=' * 60}")

    # 读取暴露数据
    exp = pd.read_csv(exposure_file, sep='\t')
    # 清洗基因名（去除 | 后面的数字）
    exp['gene'] = exp['gene'].str.split('|').str[0]
    exp = exp.rename(columns={'beta': 'beta_exp', 'se': 'se_exp', 'pval': 'pval_exp'})

    # 读取结局数据
    out = pd.read_csv(outcome_file, sep='\t')
    out = out[['SNP', 'beta', 'se', 'pval']].rename(
        columns={'beta': 'beta_out', 'se': 'se_out', 'pval': 'pval_out'})

    results_list = []
    for gene in exp['gene'].unique():
        print(f"\n  处理基因: {gene}")
        sub = exp[exp['gene'] == gene].copy()

        # 与结局合并
        merged = pd.merge(sub, out, on='SNP', how='inner')
        print(f"    匹配 SNP 数量: {len(merged)}")

        if len(merged) == 0:
            print("    无匹配 SNP，跳过")
            continue

        # 计算 F 统计量
        merged['f_stat'] = (merged['beta_exp'] / merged['se_exp']) ** 2
        mean_f = merged['f_stat'].mean()
        print(f"    平均 F 统计量: {mean_f:.2f}")

        # --- 多 SNP 使用 IVW ---
        if len(merged) >= 2:
            ivw_res = mr_ivw(
                merged,
                beta_exp_col='beta_exp',
                se_exp_col='se_exp',
                beta_out_col='beta_out',
                se_out_col='se_out'
            )
            if ivw_res:
                print(f"    IVW: beta = {ivw_res['beta']:.4f}, "
                      f"p = {ivw_res['pval']:.4e}, SNP = {ivw_res['n_snps']}")
                results_list.append({
                    'cancer_type': cancer_type,
                    'eqtl_type': eqtl_type,
                    'gene': gene,
                    'method': 'IVW',
                    'beta': ivw_res['beta'],
                    'se': ivw_res['se'],
                    'pval': ivw_res['pval'],
                    'ci_lower': ivw_res['ci_lower'],
                    'ci_upper': ivw_res['ci_upper'],
                    'n_snps': ivw_res['n_snps'],
                    'mean_f': mean_f
                })
            else:
                print("    IVW 分析失败")
        else:
            # --- 单个 SNP 使用 Wald ratio ---
            wald_res = wald_ratio_single(
                merged,
                beta_exp_col='beta_exp',
                se_exp_col='se_exp',
                beta_out_col='beta_out',
                se_out_col='se_out'
            )
            if wald_res:
                print(f"    Wald ratio: beta = {wald_res['beta']:.4f}, "
                      f"p = {wald_res['pval']:.4e}")
                results_list.append({
                    'cancer_type': cancer_type,
                    'eqtl_type': eqtl_type,
                    'gene': gene,
                    'method': 'Wald ratio',
                    'beta': wald_res['beta'],
                    'se': wald_res['se'],
                    'pval': wald_res['pval'],
                    'ci_lower': wald_res['ci_lower'],
                    'ci_upper': wald_res['ci_upper'],
                    'n_snps': 1,
                    'mean_f': mean_f
                })
            else:
                print("    Wald ratio 计算失败")

    return pd.DataFrame(results_list)


def main():
    if not OUTCOME_FILE.exists():
        print("错误: 未找到肺癌结局数据，请先运行 prepare_mr_data.py")
        return

    # 收集所有 TCGA eQTL 暴露文件
    exposure_files = list(TCGA_DIR.glob("*_cis_target_eqtl.tsv"))
    if not exposure_files:
        print("未找到暴露文件，请先运行 prepare_tcga_eqtl.py")
        return

    all_results = []
    for f in exposure_files:
        parts = f.stem.split('_')
        cancer_type = parts[0]
        eqtl_type = parts[1]
        df_res = analyze_exposure(f, OUTCOME_FILE, cancer_type, eqtl_type)
        if df_res is not None and len(df_res) > 0:
            all_results.append(df_res)

    if all_results:
        final_res = pd.concat(all_results, ignore_index=True)
        final_res.to_csv(RESULTS_DIR / "tcga_eqtl_mr_results.csv", index=False)
        print(f"\n✅ 所有结果已保存至: {RESULTS_DIR / 'tcga_eqtl_mr_results.csv'}")

        # 打印显著结果摘要
        print("\n=== 结果摘要 ===")
        sig = final_res[final_res['pval'] < 0.05]
        if len(sig) > 0:
            print("⭐ 显著结果 (p < 0.05):")
            for _, row in sig.iterrows():
                print(f"  {row['cancer_type']} {row['gene']}: "
                      f"beta={row['beta']:.4f}, p={row['pval']:.4e}, "
                      f"method={row['method']}, SNP={row['n_snps']}")
        else:
            print("⚠️ 无显著结果")

        # 打印所有结果
        print("\n📊 所有 MR 结果:")
        print(final_res[['cancer_type', 'gene', 'method', 'beta', 'se', 'pval', 'n_snps', 'mean_f']].to_string(
            index=False))
    else:
        print("❌ 无有效结果")


if __name__ == "__main__":
    main()