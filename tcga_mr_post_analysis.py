#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCGA eQTL MR 后续分析：敏感性分析、中介分析、可视化、论文表格
依赖：tcga_mr_analysis.py 生成的 TCGA_MR_Results/tcga_eqtl_mr_results.csv
      以及已有的中介数据（淋巴细胞计数等）
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ========== 配置 ==========
DATA_DIR = Path("MR_Data")
TCGA_RESULT_FILE = Path("TCGA_MR_Results/tcga_eqtl_mr_results.csv")
TCGA_EXPOSURE_DIR = Path("TCGA_eQTL_MR")
OUTCOME_FILE = DATA_DIR / "lung_cancer_outcome.tsv"
RESULTS_DIR = Path("TCGA_MR_Results")
RESULTS_DIR.mkdir(exist_ok=True)

# 中介变量文件（您已有的）
MEDIATOR_FILES = {
    'lymphocyte': DATA_DIR / "lymphocyte_mediator.tsv",
    'monocyte': DATA_DIR / "monocyte_mediator.tsv",
    'neutrophil': DATA_DIR / "neut.tsv.gz",  # 如果有
    'wbc': DATA_DIR / "wbc.tsv.gz"  # 如果有
}
# ==========================

# ---------- 1. 加载基础 MR 结果 ----------
df_mr = pd.read_csv(TCGA_RESULT_FILE)
print("已加载 TCGA MR 结果，共 {} 行".format(len(df_mr)))


# ---------- 2. 对 IL18 进行额外 MR 敏感性分析（MR-Egger, 加权中位数）----------
def mr_egger(df, beta_exp_col='beta_exp', se_exp_col='se_exp',
             beta_out_col='beta_out', se_out_col='se_out'):
    """MR-Egger 回归（加权）"""
    beta_exp = df[beta_exp_col].values
    beta_out = df[beta_out_col].values
    se_out = df[se_out_col].values
    weights = 1 / (se_out ** 2)

    from statsmodels.api import add_constant, WLS
    X = add_constant(beta_exp)
    model = WLS(beta_out, X, weights=weights)
    results = model.fit()

    intercept = results.params[0]
    slope = results.params[1]
    intercept_se = results.bse[0]
    slope_se = results.bse[1]
    intercept_p = results.pvalues[0]
    slope_p = results.pvalues[1]

    ci_lower = slope - 1.96 * slope_se
    ci_upper = slope + 1.96 * slope_se

    return {
        'method': 'MR-Egger',
        'beta': slope,
        'se': slope_se,
        'pval': slope_p,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'intercept': intercept,
        'intercept_se': intercept_se,
        'intercept_p': intercept_p,
        'n_snps': len(df)
    }


def weighted_median_bootstrap(df, beta_exp_col='beta_exp', se_exp_col='se_exp',
                              beta_out_col='beta_out', se_out_col='se_out',
                              n_bootstrap=1000):
    """
    加权中位数（Bootstrap 标准误）- 修复版
    全部使用 numpy 数组，避免 pandas 索引陷阱
    """
    # 提取列并转换为 numpy 数组
    beta_exp = df[beta_exp_col].values
    se_exp = df[se_exp_col].values
    beta_out = df[beta_out_col].values
    se_out = df[se_out_col].values

    # 过滤无效值
    valid = (beta_exp != 0) & np.isfinite(beta_exp) & np.isfinite(beta_out) & (se_exp > 0) & (se_out > 0)
    df_valid = df[valid].copy()
    if len(df_valid) < 2:
        print("   加权中位数失败: 有效 SNP 不足 2 个")
        return None

    # 提取有效值（转为 numpy 数组）
    beta_exp_v = df_valid[beta_exp_col].values
    se_exp_v = df_valid[se_exp_col].values
    beta_out_v = df_valid[beta_out_col].values
    se_out_v = df_valid[se_out_col].values

    # Wald ratio 及标准误
    wald = beta_out_v / beta_exp_v
    wald_se = np.sqrt(
        (se_out_v**2 / beta_exp_v**2) +
        (beta_out_v**2 * se_exp_v**2 / beta_exp_v**4)
    )
    weights = 1 / (wald_se ** 2)

    # --- 加权中位数 ---
    order = np.argsort(wald)
    w_sorted = weights[order]
    w_cumsum = np.cumsum(w_sorted)
    w_total = w_cumsum[-1]          # numpy 数组支持 -1 索引
    idx = np.searchsorted(w_cumsum, w_total / 2)
    if idx >= len(wald):
        idx = len(wald) - 1
    beta_wm = wald[order][idx]

    # --- Bootstrap 标准误 ---
    np.random.seed(42)
    n = len(df_valid)
    boot_betas = []
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n, size=n, replace=True)
        boot_df = df_valid.iloc[boot_idx]
        boot_wald = boot_df[beta_out_col].values / boot_df[beta_exp_col].values
        boot_betas.append(np.median(boot_wald))
    se_wm = np.std(boot_betas)
    p_wm = 2 * (1 - stats.norm.cdf(np.abs(beta_wm / se_wm))) if se_wm > 0 else 1.0
    ci_lower = beta_wm - 1.96 * se_wm
    ci_upper = beta_wm + 1.96 * se_wm

    return {
        'method': 'Weighted median',
        'beta': beta_wm,
        'se': se_wm,
        'pval': p_wm,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_snps': len(df_valid)
    }


def mr_presso_simple(df, beta_exp_col='beta_exp', se_exp_col='se_exp',
                     beta_out_col='beta_out', se_out_col='se_out'):
    """简易 MR-PRESSO（全局检验 + 离群值校正）"""
    # 计算每个 SNP 的 Wald ratio 和标准误
    beta_exp = df[beta_exp_col].values
    beta_out = df[beta_out_col].values
    se_out = df[se_out_col].values
    se_exp = df[se_exp_col].values

    valid = (beta_exp != 0) & np.isfinite(beta_exp) & np.isfinite(beta_out)
    df_valid = df[valid].copy()
    if len(df_valid) < 3:
        return None

    beta_exp_v = df_valid[beta_exp_col].values
    beta_out_v = df_valid[beta_out_col].values
    se_out_v = df_valid[se_out_col].values
    se_exp_v = df_valid[se_exp_col].values

    wald = beta_out_v / beta_exp_v
    wald_se = np.sqrt(
        (se_out_v ** 2 / beta_exp_v ** 2) +
        (beta_out_v ** 2 * se_exp_v ** 2 / beta_exp_v ** 4)
    )

    # 简单离群值检测（基于 Jackknife 残差）
    n = len(wald)
    residuals = wald - np.mean(wald)
    # 计算每个 SNP 剔除后的 IVW 估计
    ivw_drop = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w_drop = wald[mask]
        se_drop = wald_se[mask]
        weights = 1 / (se_drop ** 2)
        beta_drop = np.sum(w_drop * weights) / np.sum(weights)
        ivw_drop.append(beta_drop)
    # 比较原始 IVW 与剔除后的差异
    ivw_all = np.sum(wald * (1 / wald_se ** 2)) / np.sum(1 / wald_se ** 2)
    diff = np.abs(ivw_drop - ivw_all)
    outliers = np.where(diff > 2 * np.std(diff))[0]  # 简单阈值

    # 剔除离群值后重新 IVW
    keep = np.ones(n, dtype=bool)
    keep[outliers] = False
    if np.sum(keep) < 2:
        corrected = None
    else:
        w_keep = wald[keep]
        se_keep = wald_se[keep]
        weights_c = 1 / (se_keep ** 2)
        beta_c = np.sum(w_keep * weights_c) / np.sum(weights_c)
        se_c = np.sqrt(1 / np.sum(weights_c))
        p_c = 2 * (1 - stats.norm.cdf(np.abs(beta_c / se_c)))
        corrected = {'beta': beta_c, 'se': se_c, 'pval': p_c, 'n_snps': np.sum(keep)}

    return {
        'outliers': list(outliers),
        'corrected': corrected,
        'global_p': None  # 简化版，不计算模拟 p 值
    }


# 对 LUAD 和 LUSC 的 IL18 进行敏感性分析
sensitivity_results = []
for cancer in ['LUAD', 'LUSC']:
    # 读取原始暴露数据
    exp_file = TCGA_EXPOSURE_DIR / f"{cancer}_cis_target_eqtl.tsv"
    if not exp_file.exists():
        continue
    exp = pd.read_csv(exp_file, sep='\t')
    exp['gene'] = exp['gene'].str.split('|').str[0]
    exp = exp[exp['gene'] == 'IL18'].rename(columns={'beta': 'beta_exp', 'se': 'se_exp'})

    # 读取结局
    out = pd.read_csv(OUTCOME_FILE, sep='\t')
    out = out[['SNP', 'beta', 'se']].rename(columns={'beta': 'beta_out', 'se': 'se_out'})

    # 合并
    merged = pd.merge(exp, out, on='SNP', how='inner')
    if len(merged) < 3:
        continue

    print(f"\n--- {cancer} IL18 敏感性分析 ---")
    # 1. MR-Egger
    egger_res = mr_egger(merged)
    if egger_res:
        egger_res['cancer'] = cancer
        sensitivity_results.append(egger_res)
        print(
            f"  MR-Egger: beta={egger_res['beta']:.4f}, p={egger_res['pval']:.4e}, intercept_p={egger_res['intercept_p']:.4f}")

    # 2. 加权中位数
    wm_res = weighted_median_bootstrap(merged)
    if wm_res:
        wm_res['cancer'] = cancer
        sensitivity_results.append(wm_res)
        print(f"  Weighted median: beta={wm_res['beta']:.4f}, p={wm_res['pval']:.4e}")

    # 3. MR-PRESSO 简化版
    presso_res = mr_presso_simple(merged)
    if presso_res and presso_res['corrected']:
        print(
            f"  MR-PRESSO: detected {len(presso_res['outliers'])} outlier(s), corrected beta={presso_res['corrected']['beta']:.4f}, p={presso_res['corrected']['pval']:.4e}")

# 保存敏感性分析结果
if sensitivity_results:
    df_sens = pd.DataFrame(sensitivity_results)
    df_sens.to_csv(RESULTS_DIR / "il18_sensitivity_analysis.csv", index=False)
    print("\n敏感性分析结果已保存至:", RESULTS_DIR / "il18_sensitivity_analysis.csv")


# ---------- 3. 中介分析（IL18 eQTL → 淋巴细胞 → 肺癌）----------
def mediation_analysis_improved(exposure_df, mediator_df, outcome_df, n_bootstrap=1000):
    """
    改进的中介分析（IVW 两步法 + Bootstrap CI）
    exposure_df: 必须包含 SNP, beta_exp, se_exp
    mediator_df: 必须包含 SNP, beta_med, se_med
    outcome_df:  必须包含 SNP, beta_out, se_out
    """
    # 第一步：暴露 → 中介
    exp_med = pd.merge(exposure_df, mediator_df, on='SNP', suffixes=('_exp', '_med'), how='inner')
    exp_med = exp_med[(exp_med['beta_exp'] != 0) & (exp_med['beta_med'] != 0)]
    if len(exp_med) < 2:
        return None
    weights_a = 1 / (exp_med['se_med'] ** 2 / exp_med['beta_exp'] ** 2 +
                     exp_med['beta_med'] ** 2 * exp_med['se_exp'] ** 2 / exp_med['beta_exp'] ** 4)
    a = np.average(exp_med['beta_med'] / exp_med['beta_exp'], weights=weights_a)
    se_a = np.sqrt(1 / np.sum(weights_a))
    p_a = 2 * (1 - stats.norm.cdf(np.abs(a / se_a))) if se_a > 0 else 1.0

    # 第二步：中介 → 结局
    med_out = pd.merge(mediator_df, outcome_df, on='SNP', suffixes=('_med', '_out'), how='inner')
    med_out = med_out[(med_out['beta_med'] != 0) & (med_out['beta_out'] != 0)]
    if len(med_out) < 2:
        return None
    weights_b = 1 / (med_out['se_out'] ** 2 / med_out['beta_med'] ** 2 +
                     med_out['beta_out'] ** 2 * med_out['se_med'] ** 2 / med_out['beta_med'] ** 4)
    b = np.average(med_out['beta_out'] / med_out['beta_med'], weights=weights_b)
    se_b = np.sqrt(1 / np.sum(weights_b))
    p_b = 2 * (1 - stats.norm.cdf(np.abs(b / se_b))) if se_b > 0 else 1.0

    # 间接效应
    indirect = a * b
    # Bootstrap CI
    np.random.seed(123)
    boot_indirect = []
    for _ in range(n_bootstrap):
        # 重采样第一步
        boot_exp_med = exp_med.sample(frac=1, replace=True)
        w_a_boot = 1 / (boot_exp_med['se_med'] ** 2 / boot_exp_med['beta_exp'] ** 2 +
                        boot_exp_med['beta_med'] ** 2 * boot_exp_med['se_exp'] ** 2 / boot_exp_med['beta_exp'] ** 4)
        a_boot = np.average(boot_exp_med['beta_med'] / boot_exp_med['beta_exp'], weights=w_a_boot)
        # 重采样第二步
        boot_med_out = med_out.sample(frac=1, replace=True)
        w_b_boot = 1 / (boot_med_out['se_out'] ** 2 / boot_med_out['beta_med'] ** 2 +
                        boot_med_out['beta_out'] ** 2 * boot_med_out['se_med'] ** 2 / boot_med_out['beta_med'] ** 4)
        b_boot = np.average(boot_med_out['beta_out'] / boot_med_out['beta_med'], weights=w_b_boot)
        boot_indirect.append(a_boot * b_boot)
    ci_lower, ci_upper = np.percentile(boot_indirect, [2.5, 97.5])

    # 总效应（暴露 → 结局，使用 IVW）
    exp_out = pd.merge(exposure_df, outcome_df, on='SNP', suffixes=('_exp', '_out'), how='inner')
    exp_out = exp_out[(exp_out['beta_exp'] != 0) & (exp_out['beta_out'] != 0)]
    if len(exp_out) >= 2:
        weights_total = 1 / (exp_out['se_out'] ** 2 / exp_out['beta_exp'] ** 2 +
                             exp_out['beta_out'] ** 2 * exp_out['se_exp'] ** 2 / exp_out['beta_exp'] ** 4)
        total = np.average(exp_out['beta_out'] / exp_out['beta_exp'], weights=weights_total)
    else:
        total = np.nan

    direct = total - indirect if not np.isnan(total) else np.nan

    return {
        'a': a, 'se_a': se_a, 'p_a': p_a,
        'b': b, 'se_b': se_b, 'p_b': p_b,
        'indirect': indirect,
        'indirect_CI_lower': ci_lower,
        'indirect_CI_upper': ci_upper,
        'total_effect': total,
        'direct_effect': direct,
        'n_snps_XM': len(exp_med),
        'n_snps_MY': len(med_out)
    }


# 对 LUAD 和 LUSC 的 IL18 进行中介分析（淋巴细胞）
outcome_df = pd.read_csv(OUTCOME_FILE, sep='\t')
outcome_df = outcome_df[['SNP', 'beta', 'se']].rename(columns={'beta': 'beta_out', 'se': 'se_out'})

mediation_results = []
for cancer in ['LUAD', 'LUSC']:
    exp_file = TCGA_EXPOSURE_DIR / f"{cancer}_cis_target_eqtl.tsv"
    if not exp_file.exists():
        continue
    exp = pd.read_csv(exp_file, sep='\t')
    exp['gene'] = exp['gene'].str.split('|').str[0]
    exp = exp[exp['gene'] == 'IL18'].rename(columns={'beta': 'beta_exp', 'se': 'se_exp'})

    # 使用淋巴细胞作为中介
    if MEDIATOR_FILES['lymphocyte'].exists():
        print(f"\n--- {cancer} IL18 中介分析（淋巴细胞）---")
        med_df = pd.read_csv(MEDIATOR_FILES['lymphocyte'], sep='\t')
        med_df = med_df[['SNP', 'beta', 'se']].rename(columns={'beta': 'beta_med', 'se': 'se_med'})
        res = mediation_analysis_improved(exp, med_df, outcome_df)
        if res:
            res['cancer'] = cancer
            res['mediator'] = 'lymphocyte'
            mediation_results.append(res)
            print(f"  a (IL18->Lym): beta={res['a']:.4f}, p={res['p_a']:.4e}")
            print(f"  b (Lym->Lung): beta={res['b']:.4f}, p={res['p_b']:.4e}")
            print(
                f"  Indirect effect = {res['indirect']:.6f} (95% CI: {res['indirect_CI_lower']:.6f}, {res['indirect_CI_upper']:.6f})")
            print(f"  Proportion mediated = {res['indirect'] / res['total_effect'] * 100:.1f}%")
        else:
            print("  中介分析失败（SNP不足）")
    else:
        print("  淋巴细胞数据不存在，跳过中介分析")

if mediation_results:
    df_med = pd.DataFrame(mediation_results)
    df_med.to_csv(RESULTS_DIR / "il18_mediation_analysis.csv", index=False)
    print("\n中介分析结果已保存至:", RESULTS_DIR / "il18_mediation_analysis.csv")

# ---------- 4. 可视化 ----------
# 4.1 森林图：TCGA MR 结果
plt.figure(figsize=(8, 5))
y_pos = np.arange(len(df_mr))
plt.errorbar(df_mr['beta'], y_pos,
             xerr=[df_mr['beta'] - df_mr['ci_lower'], df_mr['ci_upper'] - df_mr['beta']],
             fmt='o', color='steelblue', ecolor='gray', capsize=3)
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
plt.yticks(y_pos, df_mr['cancer_type'] + ' ' + df_mr['gene'] + ' (' + df_mr['method'] + ')')
plt.xlabel('Effect size (β) for lung cancer')
plt.title('TCGA eQTL MR: IL18 and other NLRP3 genes')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'tcga_mr_forest.png', dpi=300)
print("\n森林图已保存至:", RESULTS_DIR / 'tcga_mr_forest.png')

# 4.2 散点图（IL18 单个 SNP 效应 vs 暴露效应）
for cancer in ['LUAD', 'LUSC']:
    exp_file = TCGA_EXPOSURE_DIR / f"{cancer}_cis_target_eqtl.tsv"
    if not exp_file.exists():
        continue
    exp = pd.read_csv(exp_file, sep='\t')
    exp['gene'] = exp['gene'].str.split('|').str[0]
    exp = exp[exp['gene'] == 'IL18'].rename(columns={'beta': 'beta_exp', 'se': 'se_exp'})
    out = pd.read_csv(OUTCOME_FILE, sep='\t')
    out = out[['SNP', 'beta', 'se']].rename(columns={'beta': 'beta_out', 'se': 'se_out'})
    merged = pd.merge(exp, out, on='SNP', how='inner')

    plt.figure(figsize=(6, 5))
    plt.scatter(merged['beta_exp'], merged['beta_out'], alpha=0.7, s=30)
    # 添加回归线（IVW 斜率）
    from scipy import stats

    slope = df_mr[(df_mr['cancer_type'] == cancer) & (df_mr['gene'] == 'IL18')]['beta'].values[0]
    x_vals = np.linspace(merged['beta_exp'].min(), merged['beta_exp'].max(), 10)
    plt.plot(x_vals, slope * x_vals, 'r--', label=f'IVW slope = {slope:.4f}')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('IL18 eQTL effect (β_exp)')
    plt.ylabel('Lung cancer effect (β_out)')
    plt.title(f'{cancer} IL18: SNP effect plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{cancer}_IL18_scatter.png', dpi=300)
    print(f"散点图已保存至: {RESULTS_DIR}/{cancer}_IL18_scatter.png")

# 4.3 漏斗图（IL18）
for cancer in ['LUAD', 'LUSC']:
    exp_file = TCGA_EXPOSURE_DIR / f"{cancer}_cis_target_eqtl.tsv"
    if not exp_file.exists():
        continue
    exp = pd.read_csv(exp_file, sep='\t')
    exp['gene'] = exp['gene'].str.split('|').str[0]
    exp = exp[exp['gene'] == 'IL18'].rename(columns={'beta': 'beta_exp', 'se': 'se_exp'})
    out = pd.read_csv(OUTCOME_FILE, sep='\t')
    out = out[['SNP', 'beta', 'se']].rename(columns={'beta': 'beta_out', 'se': 'se_out'})
    merged = pd.merge(exp, out, on='SNP', how='inner')

    wald = merged['beta_out'] / merged['beta_exp']
    se_wald = np.sqrt(
        (merged['se_out'] ** 2 / merged['beta_exp'] ** 2) +
        (merged['beta_out'] ** 2 * merged['se_exp'] ** 2 / merged['beta_exp'] ** 4)
    )
    precision = 1 / se_wald

    plt.figure(figsize=(6, 5))
    plt.scatter(wald, precision, alpha=0.7)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Wald ratio (β)')
    plt.ylabel('Precision (1/SE)')
    plt.title(f'{cancer} IL18: Funnel plot')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{cancer}_IL18_funnel.png', dpi=300)
    print(f"漏斗图已保存至: {RESULTS_DIR}/{cancer}_IL18_funnel.png")

# ---------- 5. 生成论文表格 ----------
# Table 1: TCGA eQTL MR 结果汇总
table1 = df_mr[['cancer_type', 'gene', 'method', 'n_snps', 'mean_f',
                'beta', 'se', 'pval', 'ci_lower', 'ci_upper']].copy()
table1['beta (95% CI)'] = table1.apply(
    lambda x: f"{x['beta']:.4f} ({x['ci_lower']:.4f}–{x['ci_upper']:.4f})", axis=1)
table1['p-value'] = table1['pval'].apply(lambda x: f"{x:.2e}")
table1 = table1.drop(columns=['beta', 'se', 'pval', 'ci_lower', 'ci_upper'])
table1.to_csv(RESULTS_DIR / "table1_tcga_mr_results.csv", index=False)
print("\nTable 1 已保存至:", RESULTS_DIR / "table1_tcga_mr_results.csv")

# Table 2: IL18 敏感性分析结果
if sensitivity_results:
    df_sens_table = pd.DataFrame(sensitivity_results)
    df_sens_table['beta (95% CI)'] = df_sens_table.apply(
        lambda x: f"{x['beta']:.4f} ({x['ci_lower']:.4f}–{x['ci_upper']:.4f})", axis=1)
    df_sens_table['p-value'] = df_sens_table['pval'].apply(lambda x: f"{x:.2e}")
    df_sens_table.to_csv(RESULTS_DIR / "table2_il18_sensitivity.csv", index=False)
    print("Table 2 已保存至:", RESULTS_DIR / "table2_il18_sensitivity.csv")

# Table 3: 中介分析结果
if mediation_results:
    df_med_table = pd.DataFrame(mediation_results)
    df_med_table['indirect (95% CI)'] = df_med_table.apply(
        lambda x: f"{x['indirect']:.6f} ({x['indirect_CI_lower']:.6f}–{x['indirect_CI_upper']:.6f})", axis=1)
    df_med_table['proportion mediated'] = (df_med_table['indirect'] / df_med_table['total_effect'] * 100).round(1)
    df_med_table.to_csv(RESULTS_DIR / "table3_il18_mediation.csv", index=False)
    print("Table 3 已保存至:", RESULTS_DIR / "table3_il18_mediation.csv")

print("\n" + "=" * 60)
print("所有后续分析完成！结果保存在 TCGA_MR_Results/ 目录")
print("=" * 60)