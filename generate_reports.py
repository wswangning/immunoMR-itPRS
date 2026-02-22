#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目三完整分析脚本
自动扫描当前目录下的数据文件，执行：
- 数据加载与清洗
- 工具变量筛选
- PRS构建（需PRS-CS）
- 两样本MR（免疫表型 vs 肺癌）
- 中介MR（免疫细胞 -> 细胞因子 -> 肺癌）
- 共定位与SMR（eQTL数据）
- 结果汇总与可视化
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gzip
import tarfile
from statsmodels.api import WLS, add_constant

warnings.filterwarnings('ignore')

# ==================== 通用工具函数 ====================

def detect_sep(filepath, sample_lines=5):
    """自动检测分隔符，支持 gz 文件"""
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

def clean_snp_id(snp):
    """提取纯rs号"""
    if pd.isna(snp):
        return snp
    s = str(snp).strip()
    if ':' in s:
        return s.split(':')[0]
    return s

def guess_columns(df, keywords):
    """根据关键词列表猜测列名"""
    for col in df.columns:
        col_lower = str(col).lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None

def preview_file(filepath):
    """预览文件并返回列名"""
    sep = detect_sep(filepath)
    try:
        if filepath.endswith('.gz'):
            df = pd.read_csv(filepath, sep=sep, nrows=5, compression='gzip')
        else:
            df = pd.read_csv(filepath, sep=sep, nrows=5)
        print(f"\n文件: {os.path.basename(filepath)}")
        print(f"分隔符: '{sep}'")
        print("列名:", df.columns.tolist())
        print(df.head())
        return df.columns.tolist()
    except Exception as e:
        print(f"无法预览 {filepath}: {e}")
        return []

# ==================== 数据加载类 ====================

class DataLoader:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, '*'))
        self.file_names = [os.path.basename(f) for f in self.files]
        self.data = {}  # 存储加载后的DataFrame

    def categorize_files(self):
        categories = {'immune_cell': [], 'cytokine': [], 'lung_cancer': [], 'eqtl': [], 'other': []}
        immune_keywords = ['lymph', 'neut', 'wbc', 'pdw', 'lymphocyte', 'monocyte', 'baso', 'eo']
        cytokine_keywords = ['il1b', 'il18', 'tnfa', 'il6', 'il8', 'il10', 'ifng']
        lung_keywords = ['lung', 'cancer', 'ilu', 'c3_lung', 'nsclc', 'sclc']
        eqtl_keywords = ['eqtl', 'trans', 'whole', 'gwas_eqtl', 'cis-eqtl']
        skip_ext = ('.png', '.html', '.pdf', '.jpg', '.jpeg', '.gif', '.py', '.vcf', '.vcf.gz', '.zip')
        skip_pattern = ('_results.csv', '_signif', '_summary', 'Neodymium')

        for fname in self.file_names:
            full_path = os.path.join(self.data_dir, fname)
            if os.path.isdir(full_path):
                continue  # 跳过目录
            if fname.endswith(skip_ext) or any(p in fname for p in skip_pattern):
                continue
            fname_lower = fname.lower()
            if any(kw in fname_lower for kw in immune_keywords):
                categories['immune_cell'].append(fname)
            elif any(kw in fname_lower for kw in cytokine_keywords):
                categories['cytokine'].append(fname)
            elif any(kw in fname_lower for kw in lung_keywords):
                categories['lung_cancer'].append(fname)
            elif any(kw in fname_lower for kw in eqtl_keywords):
                categories['eqtl'].append(fname)
            else:
                categories['other'].append(fname)
        return categories

    def load_gwas(self, filepath, expected_cols=None):
        """加载GWAS文件，使用文件名精确映射，若失败则跳过"""
        sep = detect_sep(filepath)
        try:
            if filepath.endswith('.gz'):
                df = pd.read_csv(filepath, sep=sep, compression='gzip', low_memory=False)
            else:
                df = pd.read_csv(filepath, sep=sep, low_memory=False)
        except Exception as e:
            print(f"  读取失败: {e}")
            return None

        fname = os.path.basename(filepath)
        # 跳过非数据文件
        if fname.endswith(('.png', '.html', '.pdf', '.jpg', '.jpeg', '.gif', '.py', '.vcf', '.vcf.gz')):
            print(f"  跳过非数据文件: {fname}")
            return None

        print(f"  文件列名: {df.columns.tolist()}")

        # 文件名到列映射的字典
        mapping = {
            # 免疫细胞 GWAS（IEU格式）
            'ieu_lymphocytes.tsv': {'snp': 'SNP', 'beta': 'beta', 'se': 'se', 'pval': 'pval', 'a1': 'effect_allele',
                                    'a2': 'other_allele'},
            'ieu_monocytes.tsv': {'snp': 'SNP', 'beta': 'beta', 'se': 'se', 'pval': 'pval', 'a1': 'effect_allele',
                                  'a2': 'other_allele'},
            'lymphocyte_count_GWAS_for_MR.tsv': {'snp': 'SNP', 'beta': 'beta', 'se': 'se', 'pval': 'pval',
                                                 'a1': 'effect_allele', 'a2': 'other_allele'},
            'lymphocytes_for_MR.tsv': {'snp': 'SNP', 'beta': 'beta', 'se': 'se', 'pval': 'pval', 'a1': 'effect_allele',
                                       'a2': 'other_allele'},
            'lung_cancer_GWAS_for_MR.tsv': {'snp': 'SNP', 'beta': 'beta', 'se': 'se', 'pval': 'pval',
                                            'a1': 'effect_allele', 'a2': 'other_allele'},
            'ukb_lung_cancer.tsv': {'snp': 'SNP', 'beta': 'beta', 'se': 'se', 'pval': 'pval', 'a1': 'effect_allele',
                                    'a2': 'other_allele'},

            # 免疫细胞 GWAS（VARIANT格式）
            'lymph.tsv.gz': {'snp': 'VARIANT', 'beta': 'EFFECT', 'se': 'SE', 'pval': 'P', 'a1': 'ALT', 'a2': 'REF'},
            'lymph_p.tsv.gz': {'snp': 'VARIANT', 'beta': 'EFFECT', 'se': 'SE', 'pval': 'P', 'a1': 'ALT', 'a2': 'REF'},
            'neut.tsv.gz': {'snp': 'VARIANT', 'beta': 'EFFECT', 'se': 'SE', 'pval': 'P', 'a1': 'ALT', 'a2': 'REF'},
            'pdw.tsv.gz': {'snp': 'VARIANT', 'beta': 'EFFECT', 'se': 'SE', 'pval': 'P', 'a1': 'ALT', 'a2': 'REF'},
            'wbc.tsv.gz': {'snp': 'VARIANT', 'beta': 'EFFECT', 'se': 'SE', 'pval': 'P', 'a1': 'ALT', 'a2': 'REF'},

            # 细胞因子 pQTL（MarkerName格式）
            'IL18.data.gz': {'snp': 'MarkerName', 'beta': 'Effect', 'se': 'StdErr', 'pval': 'P.value',
                             'a1': 'EffectAllele', 'a2': 'OtherAllele'},
            'IL1b.data.gz': {'snp': 'MarkerName', 'beta': 'Effect', 'se': 'StdErr', 'pval': 'P.value',
                             'a1': 'EffectAllele', 'a2': 'OtherAllele'},
            'TNFa.data.gz': {'snp': 'MarkerName', 'beta': 'Effect', 'se': 'StdErr', 'pval': 'P.value',
                             'a1': 'EffectAllele', 'a2': 'OtherAllele'},
            'IL18_significant.tsv': {'snp': 'MarkerName', 'beta': 'Effect', 'se': 'StdErr', 'pval': 'P.value',
                                     'a1': 'EffectAllele', 'a2': 'OtherAllele'},

            # 肺癌 GWAS（FinnGen）
            'finngen_R10_C3_LUNG_NONSMALL_EXALLC.gz': {'snp': 'rsids', 'beta': 'beta', 'se': 'sebeta', 'pval': 'pval',
                                                       'a1': 'alt', 'a2': 'ref', 'chr': '#chrom', 'pos': 'pos'},
        }

        if fname in mapping:
            cols = mapping[fname]
            snp_col = cols['snp']
            beta_col = cols['beta']
            se_col = cols['se']
            pval_col = cols['pval']
            a1_col = cols.get('a1')
            a2_col = cols.get('a2')
        else:
            # 自动猜测
            snp_col = guess_columns(df, ['snp', 'rsid', 'variant', 'marker', 'rs', 'id', 'VARIANT', 'MarkerName'])
            beta_col = guess_columns(df, ['beta', 'effect', 'logor', 'or', 'b', 'BETA', 'EFFECT'])
            se_col = guess_columns(df, ['se', 'std', 'standard_error', 'sebeta', 'SE', 'StdErr'])
            pval_col = guess_columns(df, ['pval', 'p_value', 'p', 'p-value', 'P', 'P.value'])
            a1_col = guess_columns(df, ['a1', 'allele1', 'effect_allele', 'ea', 'EffectAllele', 'ALT'])
            a2_col = guess_columns(df, ['a2', 'allele2', 'other_allele', 'oa', 'OtherAllele', 'REF'])

        if snp_col is None:
            print(f"  错误：无法识别SNP列，跳过该文件。")
            return None
        if beta_col is None:
            print(f"  错误：无法识别beta列，跳过。")
            return None
        if se_col is None:
            print(f"  错误：无法识别se列，跳过。")
            return None
        if pval_col is None:
            print(f"  错误：无法识别pval列，跳过。")
            return None

        # 重命名
        rename = {snp_col: 'SNP', beta_col: 'beta', se_col: 'se', pval_col: 'pval'}
        if a1_col and a1_col in df.columns:
            rename[a1_col] = 'A1'
        if a2_col and a2_col in df.columns:
            rename[a2_col] = 'A2'
        df.rename(columns=rename, inplace=True)

        # 确保关键列存在
        needed = ['SNP', 'beta', 'se', 'pval']
        missing = [col for col in needed if col not in df.columns]
        if missing:
            print(f"  错误：重命名后缺少列 {missing}，请检查映射。")
            return None

        # 转换数值类型（增加异常处理）
        for col in ['beta', 'se', 'pval']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except TypeError as e:
                print(f"  错误：列 {col} 无法转换为数值，实际类型: {type(df[col])}，前几行: {df[col].head().tolist()}")
                return None
        df.dropna(subset=['beta', 'se', 'pval'], inplace=True)

        df['pval'] = df['pval'].clip(lower=1e-300)
        df['SNP_clean'] = df['SNP'].apply(clean_snp_id)
        if 'A1' not in df.columns:
            df['A1'] = None
        if 'A2' not in df.columns:
            df['A2'] = None
        return df

    def load_eqtl(self, filepath):
        """加载eQTL文件，支持多种格式"""
        sep = detect_sep(filepath)
        try:
            if filepath.endswith('.gz'):
                df = pd.read_csv(filepath, sep=sep, compression='gzip', low_memory=False)
            else:
                df = pd.read_csv(filepath, sep=sep, low_memory=False)
        except Exception as e:
            print(f"读取eQTL文件 {filepath} 失败: {e}")
            return None

        fname = os.path.basename(filepath)
        print(f"  文件列名: {df.columns.tolist()}")

        # eQTL 文件列名映射
        if fname in ['LUAD_tumor.cis_eQTL.txt', 'LUSC_tumor.cis_eQTL.txt']:
            # TCGA eQTL 常见格式
            rename = {
                'SNP': 'SNP',
                'gene': 'gene',
                'beta': 'beta',
                't-stat': 'tstat',
                'p-value': 'pval'
            }
            df.rename(columns=rename, inplace=True)
            if 'se' not in df.columns and 'tstat' in df.columns and 'beta' in df.columns:
                df['se'] = abs(df['beta'] / df['tstat'])
            keep = ['SNP', 'gene', 'beta', 'se', 'pval']
        elif fname == 'whole_GWAS_eQTL_re_0711.txt':
            rename = {'SNP': 'SNP', 'gene': 'gene', 'beta': 'beta', 't-stat': 'tstat', 'p-value': 'pval'}
            df.rename(columns=rename, inplace=True)
            if 'se' not in df.columns and 'tstat' in df.columns and 'beta' in df.columns:
                df['se'] = abs(df['beta'] / df['tstat'])
            keep = ['SNP', 'gene', 'beta', 'se', 'pval']
        else:
            # 通用猜测
            snp_col = guess_columns(df, ['snp', 'rsid', 'variant'])
            gene_col = guess_columns(df, ['gene', 'gene_name', 'gene_symbol'])
            beta_col = guess_columns(df, ['beta', 'slope', 'effect'])
            se_col = guess_columns(df, ['se', 'standard_error'])
            pval_col = guess_columns(df, ['pval', 'p_value', 'p'])
            if snp_col and gene_col and beta_col and pval_col:
                rename = {snp_col: 'SNP', gene_col: 'gene', beta_col: 'beta', pval_col: 'pval'}
                if se_col:
                    rename[se_col] = 'se'
                df.rename(columns=rename, inplace=True)
                if 'se' not in df.columns:
                    tstat_col = guess_columns(df, ['t-stat', 't', 'tstat'])
                    if tstat_col:
                        df['se'] = abs(df['beta'] / df[tstat_col])
                    else:
                        print("  无法计算标准误，跳过")
                        return None
                keep = ['SNP', 'gene', 'beta', 'se', 'pval']
            else:
                print("  无法识别eQTL关键列，跳过")
                return None

        missing = [col for col in keep if col not in df.columns]
        if missing:
            print(f"  缺少列: {missing}")
            return None

        df = df[keep].dropna(subset=['beta', 'se', 'pval'])
        df['pval'] = df['pval'].clip(lower=1e-300)
        df['SNP_clean'] = df['SNP'].apply(clean_snp_id)
        return df

    def load_all(self):
        """加载所有分类好的文件"""
        categories = self.categorize_files()
        for cat, file_list in categories.items():
            if cat == 'other':
                continue
            self.data[cat] = []
            for fname in file_list:
                full_path = os.path.join(self.data_dir, fname)
                print(f"\n正在加载 {cat} 文件: {fname}")
                if cat == 'eqtl':
                    df = self.load_eqtl(full_path)
                else:
                    df = self.load_gwas(full_path)
                if df is not None:
                    self.data[cat].append({'name': fname, 'df': df})
                    print(f"  成功加载 {len(df)} 行")
                else:
                    print(f"  加载失败，跳过。")
        return self.data

# ==================== 工具变量筛选 ====================

def filter_ivs(df, p_thresh=5e-8, f_thresh=10):
    """根据P值和F统计量筛选工具变量"""
    df = df.copy()
    df['F'] = (df['beta'] / df['se']) ** 2
    df = df[(df['pval'] < p_thresh) & (df['F'] > f_thresh)]
    return df

# ==================== 孟德尔随机化函数 ====================

def harmonize(exp_df, out_df):
    """协调暴露和结局数据"""
    merged = pd.merge(exp_df, out_df, on='SNP_clean', suffixes=('_exp', '_out'))
    if 'A1_exp' not in merged.columns or 'A1_out' not in merged.columns:
        # 无等位基因信息，假设已对齐
        merged['beta_out'] = merged['beta_out']
        return merged
    # 检查等位基因方向
    mask1 = (merged['A1_exp'] == merged['A1_out']) & (merged['A2_exp'] == merged['A2_out'])
    mask2 = (merged['A1_exp'] == merged['A2_out']) & (merged['A2_exp'] == merged['A1_out'])
    merged.loc[mask2, 'beta_out'] = -merged.loc[mask2, 'beta_out']
    merged.loc[mask2, ['A1_out','A2_out']] = merged.loc[mask2, ['A2_out','A1_out']].values
    merged = merged[mask1 | mask2]
    # 排除回文SNP
    pal = ((merged['A1_exp'].isin(['A','T'])) & (merged['A2_exp'].isin(['A','T']))) | \
          ((merged['A1_exp'].isin(['C','G'])) & (merged['A2_exp'].isin(['C','G'])))
    merged = merged[~pal]
    return merged

def ivw(beta_exp, beta_out, se_out):
    w = 1 / se_out**2
    beta = np.sum(w * beta_out * beta_exp) / np.sum(w * beta_exp**2)
    se = np.sqrt(1 / np.sum(w * beta_exp**2))
    p = 2 * (1 - stats.norm.cdf(abs(beta / se)))
    q = np.sum(w * (beta_out - beta * beta_exp)**2)
    q_df = len(beta_exp) - 1
    q_p = 1 - stats.chi2.cdf(q, q_df)
    return beta, se, p, q, q_p

def mr_egger(beta_exp, beta_out, se_out):
    if len(beta_exp) < 3:
        return (np.nan,)*6
    w = 1 / se_out**2
    X = add_constant(beta_exp)
    model = WLS(beta_out, X, weights=w).fit()
    return model.params[1], model.bse[1], model.pvalues[1], model.params[0], model.bse[0], model.pvalues[0]

def weighted_median(beta_exp, beta_out, se_out, n_boot=1000):
    if len(beta_exp) < 2:
        return np.nan, np.nan, np.nan
    w = 1 / se_out**2
    ratio = beta_out / beta_exp
    idx = np.argsort(ratio)
    cumw = np.cumsum(w[idx]) / np.sum(w)
    med = ratio[idx][np.searchsorted(cumw, 0.5)]
    # bootstrap se
    boot = []
    for _ in range(n_boot):
        ii = np.random.choice(len(beta_exp), len(beta_exp), replace=True)
        r = beta_out[ii] / beta_exp[ii]
        wi = w[ii]
        order = np.argsort(r)
        cum = np.cumsum(wi[order]) / np.sum(wi)
        boot.append(r[order][np.searchsorted(cum, 0.5)])
    se = np.std(boot)
    p = 2 * (1 - stats.norm.cdf(abs(med / se)))
    return med, se, p

def two_sample_mr(exp_df, out_df):
    harmonized = harmonize(exp_df, out_df)
    if len(harmonized) == 0:
        return None
    b_exp = harmonized['beta_exp'].values
    b_out = harmonized['beta_out'].values
    se_out = harmonized['se_out'].values
    ivw_res = ivw(b_exp, b_out, se_out)
    egger_res = mr_egger(b_exp, b_out, se_out)
    wm_res = weighted_median(b_exp, b_out, se_out)
    res = {
        'IVW': {'beta': ivw_res[0], 'se': ivw_res[1], 'pval': ivw_res[2], 'Q': ivw_res[3], 'Q_pval': ivw_res[4]},
        'MR_Egger': {'beta': egger_res[0], 'se': egger_res[1], 'pval': egger_res[2],
                     'intercept': egger_res[3], 'intercept_se': egger_res[4], 'intercept_pval': egger_res[5]},
        'Weighted_median': {'beta': wm_res[0], 'se': wm_res[1], 'pval': wm_res[2]}
    }
    return res, harmonized

# ==================== 中介MR ====================

def mediation_mr(exp_df, med_df, out_df):
    """
    两步法中介孟德尔随机化，安全处理 None 返回值
    """
    # 暴露对中介的效应
    res_a = two_sample_mr(exp_df, med_df)
    if res_a is None:
        print("  暴露对中介的MR分析失败（无共同工具变量）")
        return None
    res_a_val, _ = res_a
    beta_a = res_a_val['IVW']['beta']
    se_a = res_a_val['IVW']['se']
    pval_a = res_a_val['IVW']['pval']

    # 中介对结局的效应
    res_b = two_sample_mr(med_df, out_df)
    if res_b is None:
        print("  中介对结局的MR分析失败（无共同工具变量）")
        return None
    res_b_val, _ = res_b
    beta_b = res_b_val['IVW']['beta']
    se_b = res_b_val['IVW']['se']
    pval_b = res_b_val['IVW']['pval']

    # 间接效应
    beta_ind = beta_a * beta_b
    se_ind = np.sqrt(beta_a**2 * se_b**2 + beta_b**2 * se_a**2)
    pval_ind = 2 * (1 - stats.norm.cdf(abs(beta_ind / se_ind)))

    return {
        'beta_a': beta_a, 'se_a': se_a, 'pval_a': pval_a,
        'beta_b': beta_b, 'se_b': se_b, 'pval_b': pval_b,
        'beta_ind': beta_ind, 'se_ind': se_ind, 'pval_ind': pval_ind
    }

# ==================== 共定位与SMR ====================

def approx_bf(pval, N):
    z = stats.norm.isf(pval/2)
    V = 1 / (2 * N * 0.85)  # 近似
    r = 0.0225 / (0.0225 + V)
    log10BF = 0.5 * (np.log10(1 - r) + (r * z**2) / np.log(10))
    return log10BF

def coloc_abf_simple(pval1, pval2, N1, N2):
    bf1 = [approx_bf(p, N1) for p in pval1]
    bf2 = [approx_bf(p, N2) for p in pval2]
    pc1 = 1e-4; pc2 = 1e-4; p12 = 5e-5
    log_prior = {'H0': np.log(1-pc1-pc2-p12), 'H1': np.log(pc1-p12),
                 'H2': np.log(pc2-p12), 'H3': np.log(p12), 'H4': np.log(p12)}
    log_bf1 = np.array(bf1) * np.log(10)
    log_bf2 = np.array(bf2) * np.log(10)
    log_post = pd.DataFrame({
        'H0': log_prior['H0'],
        'H1': log_prior['H1'] + log_bf1,
        'H2': log_prior['H2'] + log_bf2,
        'H3': log_prior['H3'] + log_bf1 + log_bf2,
        'H4': log_prior['H4'] + log_bf1 + log_bf2
    })
    log_sum = log_post.sum(axis=0)
    log_max = log_sum.max()
    pp = np.exp(log_sum - log_max) / np.sum(np.exp(log_sum - log_max))
    return dict(zip(['PP0','PP1','PP2','PP3','PP4'], pp))

def run_coloc(eqtl_df, gwas_df, gene_name, window=500000):
    sub = eqtl_df[eqtl_df['gene'] == gene_name].copy()
    if len(sub) == 0:
        return None
    chrom = sub['chr'].iloc[0]
    pos_min = sub['pos'].min()
    pos_max = sub['pos'].max()
    start = max(0, pos_min - window)
    end = pos_max + window
    gwas_region = gwas_df[(gwas_df['chr'] == chrom) & (gwas_df['pos'] >= start) & (gwas_df['pos'] <= end)]
    merged = pd.merge(sub, gwas_region, on='SNP_clean', suffixes=('_eqtl','_gwas'))
    if len(merged) < 5:
        return None
    N_eqtl = sub['N'].iloc[0] if 'N' in sub.columns else 500
    N_gwas = 50000  # 需根据实际修改
    res = coloc_abf_simple(merged['pval_eqtl'], merged['pval_gwas'], N_eqtl, N_gwas)
    return res, merged

def smr(eqtl_df, gwas_df, gene_name):
    sub = eqtl_df[eqtl_df['gene'] == gene_name].copy()
    if len(sub) == 0:
        return None
    top = sub.loc[sub['pval'].idxmin()]
    snp = top['SNP_clean']
    beta_e = top['beta']
    se_e = top['se']
    gwas_row = gwas_df[gwas_df['SNP_clean'] == snp]
    if len(gwas_row) == 0:
        return None
    beta_g = gwas_row.iloc[0]['beta']
    se_g = gwas_row.iloc[0]['se']
    beta_smr = beta_g / beta_e
    se_smr = np.sqrt((se_g**2)/(beta_e**2) + (beta_g**2 * se_e**2)/(beta_e**4))
    pval = 2 * (1 - stats.norm.cdf(abs(beta_smr / se_smr)))
    return {'gene': gene_name, 'SNP': snp, 'beta_smr': beta_smr, 'se_smr': se_smr, 'pval': pval}

# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("项目三完整分析脚本")
    print("=" * 60)
    loader = DataLoader('.')
    data = loader.load_all()

    # 1. 预览所有文件
    print("\n\n===== 文件分类汇总 =====")
    for cat, items in data.items():
        if items:
            print(f"\n{cat}:")
            for item in items:
                print(f"  - {item['name']} ({len(item['df'])} rows)")

    # 2. 工具变量筛选（仅对暴露和中介）
    iv_dict = {}
    for cat in ['immune_cell', 'cytokine']:
        iv_dict[cat] = []
        for item in data.get(cat, []):
            df = item['df']
            df_iv = filter_ivs(df)
            print(f"{item['name']}: 原始 {len(df)} 行, 筛选后 {len(df_iv)} 个工具变量")
            iv_dict[cat].append({'name': item['name'], 'iv': df_iv})

    # 3. 选择肺癌结局文件
    lung_files = data.get('lung_cancer', [])
    if len(lung_files) == 0:
        print("错误：未找到肺癌GWAS文件")
        return
    # 优先使用lung_cancer_GWAS_for_MR.tsv
    lung_df = None
    for item in lung_files:
        if 'lung_cancer_GWAS_for_MR' in item['name']:
            lung_df = item['df']
            break
    if lung_df is None:
        lung_df = lung_files[0]['df']  # 使用第一个
    print(f"\n使用肺癌GWAS: {lung_df.name if hasattr(lung_df, 'name') else '未知'}")

    # 4. 两样本MR：每个免疫细胞/细胞因子 vs 肺癌
    mr_results = []
    for cat in ['immune_cell', 'cytokine']:
        for item in iv_dict.get(cat, []):
            exp_name = item['name']
            exp_iv = item['iv']
            if len(exp_iv) == 0:
                print(f"  {exp_name} 无工具变量，跳过")
                continue
            result = two_sample_mr(exp_iv, lung_df)
            if result is not None:
                res, harmonized = result
                mr_results.append({
                    'exposure': exp_name,
                    'category': cat,
                    'n_iv': len(harmonized),
                    'IVW_beta': res['IVW']['beta'],
                    'IVW_pval': res['IVW']['pval'],
                    'MR_Egger_beta': res['MR_Egger']['beta'],
                    'MR_Egger_pval': res['MR_Egger']['pval'],
                    'Weighted_median_beta': res['Weighted_median']['beta'],
                    'Weighted_median_pval': res['Weighted_median']['pval']
                })
                print(f"  {exp_name} vs 肺癌: IVW β={res['IVW']['beta']:.3f}, p={res['IVW']['pval']:.3e}")
            else:
                print(f"  {exp_name} vs 肺癌: MR分析失败（无工具变量或协调失败）")

    mr_df = pd.DataFrame(mr_results)
    mr_df.to_csv('mr_results.csv', index=False)
    print("\n两样本MR结果已保存至 mr_results.csv")

    # 5. 中介MR：免疫细胞 -> 细胞因子 -> 肺癌
    # 选择淋巴细胞和IL18作为示例
    lymph_file = None
    il18_file = None
    for item in iv_dict.get('immune_cell', []):
        if 'lymph' in item['name'].lower() and len(item['iv']) > 0:
            lymph_file = item['iv']
            break
    for item in iv_dict.get('cytokine', []):
        if 'il18' in item['name'].lower() and len(item['iv']) > 0:
            il18_file = item['iv']
            break

    if lymph_file is not None and il18_file is not None and lung_df is not None:
        print("\n尝试中介MR: 淋巴细胞 -> IL18 -> 肺癌")
        # 先检查淋巴细胞与IL18是否有共同工具变量（可选）
        check = two_sample_mr(lymph_file, il18_file)
        if check is None:
            print("  淋巴细胞与IL18无共同工具变量，无法进行中介MR")
        else:
            med_res = mediation_mr(lymph_file, il18_file, lung_df)
            if med_res is not None:
                print("中介MR结果（淋巴细胞 -> IL18 -> 肺癌）:")
                print(f"  β_a = {med_res['beta_a']:.3f}, p_a = {med_res['pval_a']:.3e}")
                print(f"  β_b = {med_res['beta_b']:.3f}, p_b = {med_res['pval_b']:.3e}")
                print(f"  间接效应 = {med_res['beta_ind']:.3f}, p = {med_res['pval_ind']:.3e}")
                pd.DataFrame([med_res]).to_csv('mediation_result.csv', index=False)
            else:
                print("  中介MR分析失败（函数返回None）")
    else:
        print("\n未找到可用的淋巴细胞或IL18数据，跳过中介MR")

    # 6. 共定位与SMR（使用eQTL数据）
    eqtl_files = data.get('eqtl', [])
    if len(eqtl_files) > 0 and lung_df is not None:
        # 选择第一个eQTL文件（如whole_GWAS_eQTL）
        eqtl_df = eqtl_files[0]['df']
        # 基因列表
        genes = ['IL18', 'NLRP3', 'GPX4', 'SLC7A11', 'IL1B', 'TNF', 'IFNG', 'NFE2L2']
        coloc_results = []
        smr_results = []
        for gene in genes:
            coloc_res = run_coloc(eqtl_df, lung_df, gene)
            if coloc_res:
                coloc_results.append({'gene': gene, 'PP4': coloc_res[0]['PP4'], 'n_snp': len(coloc_res[1])})
            smr_res = smr(eqtl_df, lung_df, gene)
            if smr_res:
                smr_results.append(smr_res)
        coloc_df = pd.DataFrame(coloc_results)
        smr_df = pd.DataFrame(smr_results)
        coloc_df.to_csv('coloc_results.csv', index=False)
        smr_df.to_csv('smr_results.csv', index=False)
        print("\n共定位与SMR结果已保存。")

    # 7. 可视化（简略）
    if len(mr_df) > 0:
        plt.figure(figsize=(10, 6))
        # 处理 p 值为 0 的情况（替换为极小值）
        mr_df['plot_pval'] = mr_df['IVW_pval'].clip(lower=1e-300)
        plt.scatter(mr_df['IVW_beta'], -np.log10(mr_df['plot_pval']), s=80, alpha=0.7)
        plt.xlabel('IVW beta (causal effect)')
        plt.ylabel('-log10(IVW p-value)')
        plt.title('Two-sample Mendelian Randomization Results\n(Immune traits on lung cancer)')
        # 添加标签
        for i, row in mr_df.iterrows():
            plt.annotate(row['exposure'],
                         (row['IVW_beta'], -np.log10(row['plot_pval'])),
                         fontsize=9,
                         xytext=(5, 5),
                         textcoords='offset points')
        plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--', label='p = 0.05')
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig('mr_volcano.png', dpi=150)
        print("MR火山图已保存。")

    print("\n所有分析完成！")

if __name__ == '__main__':
    main()