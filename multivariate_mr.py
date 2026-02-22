#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版多变量孟德尔随机化分析
基于已清洗的免疫细胞GWAS和肺癌GWAS，构建共同工具变量并执行多变量IVW
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import subprocess
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 是否进行LD clumping（True/False）
DO_CLUMPING = False  # 设为False则使用P<5e-8且F>10的SNP作为工具变量（无LD筛选）
PLINK_PATH = 'plink'  # PLINK可执行文件路径
REF_PREFIX = 'ld_ref/EUR/1000G_EUR'  # 参考面板前缀（不含扩展名）

# 文件路径配置（请根据实际路径修改）
DATA_DIR = '.'  # 数据所在目录
MR_RESULTS_FILE = 'mr_results.csv'  # 之前生成的mr_results.csv
OUTCOME_FILE = 'finngen_R10_C3_LUNG_NONSMALL_EXALLC.gz'  # 肺癌GWAS文件


# ==================== 工具函数 ====================

def detect_sep(filepath, sample_lines=5):
    """自动检测分隔符"""
    import gzip
    try:
        if filepath.endswith('.gz'):
            f = gzip.open(filepath, 'rt')
        else:
            f = open(filepath, 'r')
        lines = [f.readline().strip() for _ in range(sample_lines)]
        f.close()
    except Exception as e:
        print(f"无法读取文件 {filepath}: {e}")
        return '\t'
    sep_candidates = [',', '\t', ' ']
    for sep in sep_candidates:
        if all(sep in line for line in lines if line):
            return sep
    return '\t'


def load_gwas(filepath, snp_col='SNP', beta_col='beta', se_col='se',
              pval_col='pval', a1_col=None, a2_col=None):
    """加载GWAS文件，返回标准化DataFrame"""
    sep = detect_sep(filepath)
    if filepath.endswith('.gz'):
        df = pd.read_csv(filepath, sep=sep, compression='gzip', low_memory=False)
    else:
        df = pd.read_csv(filepath, sep=sep, low_memory=False)

    # 重命名列
    rename_dict = {}
    if snp_col in df.columns:
        rename_dict[snp_col] = 'SNP'
    if beta_col in df.columns:
        rename_dict[beta_col] = 'beta'
    if se_col in df.columns:
        rename_dict[se_col] = 'se'
    if pval_col in df.columns:
        rename_dict[pval_col] = 'pval'
    if a1_col and a1_col in df.columns:
        rename_dict[a1_col] = 'A1'
    if a2_col and a2_col in df.columns:
        rename_dict[a2_col] = 'A2'
    df.rename(columns=rename_dict, inplace=True)

    # 确保必需列存在
    required = ['SNP', 'beta', 'se', 'pval']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"缺少必需列: {col}")

    df['SNP_clean'] = df['SNP'].astype(str).str.split(':').str[0]  # 提取rs号
    return df


def filter_ivs(df, p_thresh=5e-8, f_thresh=10):
    """基于P值和F统计量筛选工具变量"""
    df = df.copy()
    df['F'] = (df['beta'] / df['se']) ** 2
    df = df[(df['pval'] < p_thresh) & (df['F'] > f_thresh)]
    return df


def ld_clumping(df, snp_col='SNP_clean', pval_col='pval',
                clump_p1=5e-8, clump_r2=0.001, clump_kb=10000,
                plink_path='plink', ref_prefix='ld_ref/EUR/1000G_EUR'):
    """使用PLINK进行LD clumping"""
    temp_input = 'temp_clump_input.txt'
    df[[snp_col, pval_col]].to_csv(temp_input, sep='\t', index=False, header=False)
    output_prefix = 'temp_clump_out'
    cmd = f"{plink_path} --bfile {ref_prefix} --clump {temp_input} " \
          f"--clump-p1 {clump_p1} --clump-r2 {clump_r2} --clump-kb {clump_kb} " \
          f"--out {output_prefix}"
    print(f"运行LD clumping命令: {cmd}")
    subprocess.run(cmd, shell=True)

    clumped_file = f"{output_prefix}.clumped"
    if not os.path.exists(clumped_file):
        print("LD clumping失败，请检查PLINK和参考面板")
        return []
    clumped = pd.read_csv(clumped_file, delim_whitespace=True)
    snp_list = clumped['SNP'].tolist()
    # 清理临时文件
    os.remove(temp_input)
    os.remove(clumped_file)
    os.remove(f"{output_prefix}.log")
    return snp_list


def multivariate_ivw(beta_exp, beta_out, se_out):
    """
    多变量IVW（固定效应）
    beta_exp: (n_snps, n_exposures) 数组
    beta_out: (n_snps,) 数组
    se_out: (n_snps,) 数组
    """
    W = np.diag(1 / (se_out ** 2))
    X = beta_exp
    Y = beta_out
    XtWX = X.T @ W @ X
    XtWY = X.T @ W @ Y
    try:
        beta_mv = np.linalg.solve(XtWX, XtWY)
    except np.linalg.LinAlgError:
        beta_mv = np.linalg.pinv(XtWX) @ XtWY
    resid = Y - X @ beta_mv
    sigma2 = (resid.T @ W @ resid) / (len(Y) - X.shape[1])
    var_beta = np.linalg.inv(XtWX) * sigma2
    se_mv = np.sqrt(np.diag(var_beta))
    p_mv = 2 * (1 - stats.norm.cdf(np.abs(beta_mv / se_mv)))
    return beta_mv, se_mv, p_mv


# ==================== 主流程 ====================

def main():
    # 1. 读取mr_results.csv，获取成功分析的暴露名称
    mr_df = pd.read_csv(MR_RESULTS_FILE)
    exposures = mr_df['exposure'].tolist()
    print(f"从{MR_RESULTS_FILE}中读取到{len(exposures)}个暴露: {exposures}")

    # 2. 加载肺癌结局数据（指定FinnGen的正确列名）
    print("\n加载肺癌结局数据...")
    lung_df = load_gwas(OUTCOME_FILE,
                        snp_col='rsids',  # SNP列名
                        beta_col='beta',  # 效应列名
                        se_col='sebeta',  # 标准误列名
                        pval_col='pval')  # P值列名
    lung_df = lung_df[['SNP_clean', 'beta', 'se', 'pval']].drop_duplicates('SNP_clean')
    print(f"结局数据包含 {len(lung_df)} 个SNP")

    # 3. 为每个暴露加载数据并筛选工具变量
    iv_dict = {}
    for exp_name in exposures:
        filepath = os.path.join(DATA_DIR, exp_name)
        if not os.path.exists(filepath):
            print(f"警告: 文件 {filepath} 不存在，跳过")
            continue
        print(f"\n处理暴露: {exp_name}")
        df = load_gwas(filepath)  # 免疫细胞文件通常有标准列名，无需特别指定
        # 筛选工具变量
        df_iv = filter_ivs(df)
        if len(df_iv) == 0:
            print(f"  无工具变量，跳过")
            continue
        if DO_CLUMPING:
            snp_list = ld_clumping(df_iv, plink_path=PLINK_PATH, ref_prefix=REF_PREFIX)
            df_iv = df_iv[df_iv['SNP_clean'].isin(snp_list)]
            print(f"  LD clumping后剩余 {len(df_iv)} 个工具变量")
        else:
            print(f"  筛选后 {len(df_iv)} 个工具变量（未进行LD clumping）")
        iv_dict[exp_name] = df_iv[['SNP_clean', 'beta', 'se']].rename(
            columns={'beta': f'beta_{exp_name}', 'se': f'se_{exp_name}'})

    if len(iv_dict) == 0:
        print("没有可用的暴露，退出")
        return

    # 4. 构建共同工具变量（所有暴露和结局的交集）
    all_snps = set()
    for df in iv_dict.values():
        all_snps.update(df['SNP_clean'])
    all_snps.update(lung_df['SNP_clean'])
    all_snps = list(all_snps)
    print(f"\n总SNP数（并集）: {len(all_snps)}")

    merged_df = pd.DataFrame({'SNP_clean': all_snps})
    for exp_name, df_iv in iv_dict.items():
        merged_df = merged_df.merge(df_iv, on='SNP_clean', how='left')
    merged_df = merged_df.merge(lung_df[['SNP_clean', 'beta', 'se']],
                                on='SNP_clean', how='left', suffixes=('', '_out'))
    merged_df.rename(columns={'beta': 'beta_out', 'se': 'se_out'}, inplace=True)

    cols_needed = [f'beta_{exp}' for exp in iv_dict.keys()] + ['beta_out', 'se_out']
    merged_df = merged_df.dropna(subset=cols_needed)
    print(f"共同SNP（交集）: {len(merged_df)}")

    if len(merged_df) == 0:
        print("没有共同SNP，无法进行多变量MR")
        return

    # 5. 构建多变量IVW输入矩阵
    beta_exp = merged_df[[f'beta_{exp}' for exp in iv_dict.keys()]].values
    beta_out = merged_df['beta_out'].values
    se_out = merged_df['se_out'].values

    # 6. 执行多变量IVW
    beta_mv, se_mv, p_mv = multivariate_ivw(beta_exp, beta_out, se_out)

    # 7. 输出结果
    result_df = pd.DataFrame({
        'exposure': list(iv_dict.keys()),
        'beta_mv': beta_mv,
        'se_mv': se_mv,
        'p_mv': p_mv
    })
    print("\n多变量MR结果：")
    print(result_df.to_string(index=False))

    # 保存结果
    result_df.to_csv('multivariate_mr_results.csv', index=False)
    print("\n结果已保存至 multivariate_mr_results.csv")


if __name__ == '__main__':
    main()