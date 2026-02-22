"""
项目三：基于机器学习与孟德尔随机化的免疫毒性表型与健康结局关联性研究
修正版 v2.0
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.sparse.linalg import svds
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api import WLS, add_constant
import warnings
warnings.filterwarnings('ignore')

# ============================
# 1. 数据加载与预处理
# ============================

def load_gwas_sumstats(filepath, sep='\t', snp_col='SNP', effect_col='BETA',
                        se_col='SE', pval_col='P', effect_allele='A1',
                        other_allele='A2', freq_col='MAF'):
    """
    加载GWAS汇总统计，标准化列名
    """
    df = pd.read_csv(filepath, sep=sep)
    # 重命名列为统一名称
    rename_dict = {
        snp_col: 'SNP',
        effect_col: 'BETA',
        se_col: 'SE',
        pval_col: 'P',
        effect_allele: 'A1',
        other_allele: 'A2',
        freq_col: 'MAF'
    }
    df.rename(columns=rename_dict, inplace=True)
    # 保留必要列
    keep_cols = ['SNP', 'BETA', 'SE', 'P', 'A1', 'A2', 'MAF']
    df = df[[c for c in keep_cols if c in df.columns]]
    return df

def calc_f_stat(beta, se, n):
    """计算单个SNP的F统计量"""
    return (beta / se) ** 2

# ============================
# 2. 孟德尔随机化分析（修正版）
# ============================

def harmonize_data(exp_dat, out_dat):
    """
    协调暴露与结局数据：对齐等位基因，去除回文SNP
    exp_dat: 暴露GWAS，列：SNP, BETA, SE, A1, A2
    out_dat: 结局GWAS，列：SNP, BETA, SE, A1, A2
    返回协调后的数据框
    """
    merged = pd.merge(exp_dat, out_dat, on='SNP', suffixes=('_exp', '_out'))
    # 检查等位基因是否匹配
    # 假设A1是效应等位基因，需要确保A1_exp == A1_out
    # 若A1_exp == A2_out 且 A2_exp == A1_out，则翻转效应方向
    mask1 = (merged['A1_exp'] == merged['A1_out']) & (merged['A2_exp'] == merged['A2_out'])
    mask2 = (merged['A1_exp'] == merged['A2_out']) & (merged['A2_exp'] == merged['A1_out'])
    merged.loc[mask2, 'BETA_out'] = -merged.loc[mask2, 'BETA_out']
    merged.loc[mask2, ['A1_out', 'A2_out']] = merged.loc[mask2, ['A2_out', 'A1_out']].values
    merged = merged[mask1 | mask2]  # 保留对齐的
    # 排除回文SNP（A/T或C/G）如果无法确定链向，这里简单排除
    palindromic = ((merged['A1_exp'].isin(['A','T'])) & (merged['A2_exp'].isin(['A','T']))) | \
                   ((merged['A1_exp'].isin(['C','G'])) & (merged['A2_exp'].isin(['C','G'])))
    merged = merged[~palindromic]
    return merged

def ivw(beta_exp, beta_out, se_out):
    """
    逆方差加权（IVW）固定效应
    返回：beta, se, pval, Q统计量
    """
    weights = 1 / (se_out ** 2)
    beta_ivw = np.sum(weights * beta_out * beta_exp) / np.sum(weights * beta_exp ** 2)
    se_ivw = np.sqrt(1 / np.sum(weights * beta_exp ** 2))
    pval = 2 * (1 - stats.norm.cdf(abs(beta_ivw / se_ivw)))
    # Q统计量
    q = np.sum(weights * (beta_out - beta_ivw * beta_exp) ** 2)
    q_df = len(beta_exp) - 1
    q_pval = 1 - stats.chi2.cdf(q, q_df)
    return beta_ivw, se_ivw, pval, q, q_pval

def mr_egger(beta_exp, beta_out, se_out):
    """
    MR-Egger 回归（加权最小二乘）
    要求至少3个工具变量
    """
    if len(beta_exp) < 3:
        return (np.nan,) * 6
    weights = 1 / (se_out ** 2)
    X = add_constant(beta_exp)
    model = WLS(beta_out, X, weights=weights).fit()
    beta0 = model.params[0]  # 截距（多效性）
    beta1 = model.params[1]  # 因果估计
    se0 = model.bse[0]
    se1 = model.bse[1]
    pval0 = model.pvalues[0]
    pval1 = model.pvalues[1]
    return beta1, se1, pval1, beta0, se0, pval0

def weighted_median(beta_exp, beta_out, se_out, n_boot=1000):
    """
    加权中位数估计（要求至少2个工具变量）
    """
    if len(beta_exp) < 2:
        return (np.nan,) * 3
    weights = 1 / (se_out ** 2)
    ratio = beta_out / beta_exp
    order = np.argsort(ratio)
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    median_idx = np.searchsorted(cum_weights, 0.5)
    beta_med = ratio[order][median_idx]
    # Bootstrap标准误
    boot_ests = []
    n_snp = len(beta_exp)
    for _ in range(n_boot):
        idx = np.random.choice(n_snp, n_snp, replace=True)
        ratio_boot = beta_out[idx] / beta_exp[idx]
        weights_boot = weights[idx]
        order_boot = np.argsort(ratio_boot)
        cum_boot = np.cumsum(weights_boot[order_boot]) / np.sum(weights_boot)
        median_boot = ratio_boot[order_boot][np.searchsorted(cum_boot, 0.5)]
        boot_ests.append(median_boot)
    se_med = np.std(boot_ests)
    pval = 2 * (1 - stats.norm.cdf(abs(beta_med / se_med)))
    return beta_med, se_med, pval

def two_sample_mr(exp_dat, out_dat):
    """
    完整的两个样本MR分析
    """
    harmonized = harmonize_data(exp_dat, out_dat)
    if len(harmonized) == 0:
        print("警告：协调后无SNP剩余，无法进行MR分析")
        return None

    beta_exp = harmonized['BETA_exp'].values
    beta_out = harmonized['BETA_out'].values
    se_out = harmonized['SE_out'].values

    # IVW
    ivw_res = ivw(beta_exp, beta_out, se_out)

    # MR-Egger
    egger_res = mr_egger(beta_exp, beta_out, se_out)

    # 加权中位数
    wm_res = weighted_median(beta_exp, beta_out, se_out)

    res = {
        'IVW': {'beta': ivw_res[0], 'se': ivw_res[1], 'pval': ivw_res[2], 'Q': ivw_res[3], 'Q_pval': ivw_res[4]},
        'MR_Egger': {'beta': egger_res[0], 'se': egger_res[1], 'pval': egger_res[2],
                     'intercept': egger_res[3], 'intercept_se': egger_res[4], 'intercept_pval': egger_res[5]},
        'Weighted_median': {'beta': wm_res[0], 'se': wm_res[1], 'pval': wm_res[2]}
    }
    return res

def mediation_mr(exposure_dat, mediator_dat, outcome_dat):
    """
    两步法中介孟德尔随机化
    """
    res_a = two_sample_mr(exposure_dat, mediator_dat)
    if res_a is None:
        return None
    beta_a = res_a['IVW']['beta']
    se_a = res_a['IVW']['se']

    res_b = two_sample_mr(mediator_dat, outcome_dat)
    if res_b is None:
        return None
    beta_b = res_b['IVW']['beta']
    se_b = res_b['IVW']['se']

    beta_indirect = beta_a * beta_b
    se_indirect = np.sqrt(beta_a**2 * se_b**2 + beta_b**2 * se_a**2)
    pval = 2 * (1 - stats.norm.cdf(abs(beta_indirect / se_indirect)))

    return {
        'beta_a': beta_a, 'se_a': se_a,
        'beta_b': beta_b, 'se_b': se_b,
        'beta_indirect': beta_indirect, 'se_indirect': se_indirect, 'pval': pval
    }

# ============================
# 3. 主流程示例（使用改进的模拟数据）
# ============================

if __name__ == "__main__":
    np.random.seed(123)

    # 模拟免疫细胞GWAS（暴露）——确保有足够显著的SNP
    n_snp_exp = 500
    exp_data = pd.DataFrame({
        'SNP': [f'rs{i}' for i in range(n_snp_exp)],
        'BETA': np.random.normal(0, 0.1, n_snp_exp),
        'SE': np.random.uniform(0.02, 0.05, n_snp_exp),
        'P': np.random.uniform(0, 1, n_snp_exp),
        'A1': np.random.choice(['A','C','G','T'], n_snp_exp),
        'A2': np.random.choice(['A','C','G','T'], n_snp_exp)
    })
    # 生成50个显著SNP（P < 5e-8）
    sig_idx = np.random.choice(n_snp_exp, 50, replace=False)
    exp_data.loc[sig_idx, 'P'] = np.random.uniform(1e-12, 5e-8, 50)
    exp_data.loc[sig_idx, 'BETA'] = np.random.normal(0.2, 0.05, 50)  # 赋予较大效应

    # 模拟肺癌GWAS（结局）
    out_data = pd.DataFrame({
        'SNP': exp_data['SNP'].tolist() + [f'rs{500+i}' for i in range(100)],
        'BETA': np.random.normal(0, 0.1, 600),
        'SE': np.random.uniform(0.02, 0.05, 600),
        'P': np.random.uniform(0, 1, 600),
        'A1': np.random.choice(['A','C','G','T'], 600),
        'A2': np.random.choice(['A','C','G','T'], 600)
    })

    # 步骤1：工具变量筛选（clumping + F > 10）
    exp_data['F'] = calc_f_stat(exp_data['BETA'], exp_data['SE'], n=50000)
    ivs = exp_data[(exp_data['P'] < 5e-8) & (exp_data['F'] > 10)].copy()
    print(f"筛选到 {len(ivs)} 个工具变量")

    # 若有多个工具变量，取前30个最显著的（模拟特征筛选）
    if len(ivs) > 30:
        ivs = ivs.nsmallest(30, 'P')
    print(f"最终使用 {len(ivs)} 个工具变量进行MR分析")

    # 步骤2：两样本MR分析
    res_mr = two_sample_mr(ivs, out_data)
    if res_mr:
        print("\n两样本MR结果：")
        for method, est in res_mr.items():
            if np.isnan(est['beta']):
                print(f"{method}: 工具变量数量不足，无法计算")
            else:
                print(f"{method}: beta={est['beta']:.3f}, se={est['se']:.3f}, pval={est['pval']:.3e}")

    # 步骤3：中介MR（模拟细胞因子pQTL数据）
    mediator_data = pd.DataFrame({
        'SNP': exp_data['SNP'].tolist(),
        'BETA': np.random.normal(0, 0.1, n_snp_exp),
        'SE': np.random.uniform(0.02, 0.05, n_snp_exp),
        'P': np.random.uniform(0, 1, n_snp_exp),
        'A1': exp_data['A1'],
        'A2': exp_data['A2']
    })
    # 确保部分SNP显著
    mediator_data.loc[sig_idx, 'P'] = np.random.uniform(1e-10, 5e-8, 50)
    mediator_data.loc[sig_idx, 'BETA'] = np.random.normal(0.15, 0.05, 50)

    res_med = mediation_mr(ivs, mediator_data, out_data)
    if res_med:
        print("\n中介MR结果：")
        print(f"间接效应 = {res_med['beta_indirect']:.3f}, "
              f"95% CI = [{res_med['beta_indirect']-1.96*res_med['se_indirect']:.3f}, "
              f"{res_med['beta_indirect']+1.96*res_med['se_indirect']:.3f}], "
              f"p={res_med['pval']:.3e}")

    # 可视化森林图
    if res_mr:
        fig, ax = plt.subplots(figsize=(6,4))
        methods = []
        betas = []
        ses = []
        for m, est in res_mr.items():
            if not np.isnan(est['beta']):
                methods.append(m)
                betas.append(est['beta'])
                ses.append(est['se'])
        y_pos = np.arange(len(methods))
        ax.errorbar(betas, y_pos, xerr=1.96*np.array(ses), fmt='o', capsize=5)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        ax.set_xlabel('Causal estimate (95% CI)')
        plt.tight_layout()
        plt.show()