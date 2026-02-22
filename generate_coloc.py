import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

def approx_bf(pval, N):
    z = stats.norm.isf(pval/2)
    V = 1 / (2 * N * 0.85)
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
    gene_chr = str(sub['chr'].iloc[0]).replace('chr', '')
    pos_min = sub['pos'].min()
    pos_max = sub['pos'].max()
    start = max(0, pos_min - window)
    end = pos_max + window
    gwas_region = gwas_df[(gwas_df['chr'] == gene_chr) & (gwas_df['pos'] >= start) & (gwas_df['pos'] <= end)]
    merged = pd.merge(sub, gwas_region, left_on='SNP_clean', right_on='SNP_clean', suffixes=('_eqtl', '_gwas'))
    if len(merged) < 5:
        return None
    N_eqtl = 500   # 根据实际样本量修改
    N_gwas = 50000 # 根据实际样本量修改
    res = coloc_abf_simple(merged['pval_eqtl'], merged['pval_gwas'], N_eqtl, N_gwas)
    return res, merged

# 加载 eQTL 数据
eqtl_file = 'whole_GWAS_eQTL_re_0711.txt'
print("加载 eQTL 数据...")
eqtl_df = pd.read_csv(eqtl_file, sep='\t')
print("eQTL 列名:", eqtl_df.columns.tolist())
eqtl_df.rename(columns={
    'SNP': 'SNP',
    'gene': 'gene',
    'beta': 'beta',
    't-stat': 'tstat',
    'p-value': 'pval',
    's_chr': 'chr',
    's_pos': 'pos'
}, inplace=True)
if 'se' not in eqtl_df.columns and 'tstat' in eqtl_df.columns:
    eqtl_df['se'] = abs(eqtl_df['beta'] / eqtl_df['tstat'])
eqtl_df['SNP_clean'] = eqtl_df['SNP'].astype(str).str.split(':').str[0]
eqtl_df['chr'] = eqtl_df['chr'].astype(str).str.replace('chr', '')

# 加载肺癌 GWAS 数据
lung_file = 'finngen_R10_C3_LUNG_NONSMALL_EXALLC.gz'
print("加载肺癌数据...")
gwas_df = pd.read_csv(lung_file, sep='\t', compression='gzip')
print("肺癌列名:", gwas_df.columns.tolist())
gwas_df.rename(columns={
    '#chrom': 'chr',
    'pos': 'pos',
    'rsids': 'SNP',
    'beta': 'beta',
    'sebeta': 'se',
    'pval': 'pval'
}, inplace=True)
gwas_df['SNP_clean'] = gwas_df['SNP'].astype(str).str.split(':').str[0]
gwas_df['chr'] = gwas_df['chr'].astype(str).str.replace('chr', '')

genes = ['IL18', 'NLRP3', 'GPX4', 'SLC7A11', 'IL1B', 'TNF', 'IFNG', 'NFE2L2']
results = []
for gene in genes:
    print(f"处理基因 {gene}...")
    coloc_res = run_coloc(eqtl_df, gwas_df, gene)
    if coloc_res:
        res, merged = coloc_res
        results.append({'gene': gene, 'PP4': res['PP4'], 'n_snp': len(merged)})
    else:
        results.append({'gene': gene, 'PP4': None, 'n_snp': 0})

coloc_df = pd.DataFrame(results)
coloc_df.to_csv('coloc_results.csv', index=False)
print("\n共定位结果：")
print(coloc_df)