"""
eQTL 数据处理与共定位分析模块（适配您的文件格式）
功能：
- 读取四个文件，支持列名或列索引指定列
- 自动计算标准误（如果提供了 t-stat）
- 提取特定基因或区域
- 近似贝叶斯共定位
- 简化 SMR 分析
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# ======================== 1. 数据读取与清洗 ========================

def detect_sep(filepath, sample_lines=5):
    with open(filepath, 'r') as f:
        lines = [f.readline().strip() for _ in range(sample_lines)]
    sep_candidates = [',', '\t', ' ']
    for sep in sep_candidates:
        if all(sep in line for line in lines if line):
            return sep
    return '\t'

def read_eqtl(filepath,
              snp_col=None, gene_col=None, beta_col=None, se_col=None, pval_col=None,
              tstat_col=None, maf_col=None, chr_col=None, pos_col=None,
              sep=None, has_header=True, N=None):
    if sep is None:
        sep = detect_sep(filepath)
    if has_header:
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
    else:
        df = pd.read_csv(filepath, sep=sep, header=None, low_memory=False)

    def colname(x):
        if isinstance(x, int):
            return df.columns[x]
        return x

    rename_dict = {}
    if snp_col is not None:
        rename_dict[colname(snp_col)] = 'SNP'
    if gene_col is not None:
        rename_dict[colname(gene_col)] = 'gene'
    if beta_col is not None:
        rename_dict[colname(beta_col)] = 'beta'
    if pval_col is not None:
        rename_dict[colname(pval_col)] = 'pval'
    if maf_col is not None:
        rename_dict[colname(maf_col)] = 'maf'
    if chr_col is not None:
        rename_dict[colname(chr_col)] = 'chr'
    if pos_col is not None:
        rename_dict[colname(pos_col)] = 'pos'

    if se_col is not None:
        rename_dict[colname(se_col)] = 'se'
    elif tstat_col is not None and beta_col is not None:
        tstat_name = colname(tstat_col)
        beta_name = colname(beta_col)
        df.rename(columns={beta_name: 'beta'}, inplace=True)
        df['se'] = abs(df['beta'] / df[tstat_name])
    else:
        raise ValueError("必须提供 se_col 或 tstat_col 以计算标准误")

    df.rename(columns=rename_dict, inplace=True)

    keep_cols = ['SNP', 'beta', 'se', 'pval']
    if 'gene' in df.columns:
        keep_cols.append('gene')
    if 'maf' in df.columns:
        keep_cols.append('maf')
    if 'chr' in df.columns and 'pos' in df.columns:
        keep_cols.extend(['chr','pos'])

    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df.dropna(subset=['beta','se','pval'], inplace=True)
    df['pval'] = df['pval'].clip(lower=1e-300)
    if N is not None:
        df['N'] = N
    return df

def extract_gene(df, gene_name):
    if 'gene' not in df.columns:
        raise ValueError("DataFrame 缺少 'gene' 列")
    return df[df['gene'].astype(str).str.upper() == gene_name.upper()].copy()

def extract_region(df, chrom, start, end):
    required = ['chr','pos']
    if not all(col in df.columns for col in required):
        raise ValueError("需要 'chr' 和 'pos' 列进行区域提取")
    chrom = str(chrom).replace('chr','')
    df['chr'] = df['chr'].astype(str).str.replace('chr','')
    mask = (df['chr'] == chrom) & (df['pos'].between(start, end))
    return df[mask].copy()

# ======================== 2. 共定位分析（近似贝叶斯）==================

def approx_bf(pval, N, sigma_sq=0.15):
    z = stats.norm.ppf(pval/2, lower_tail=False)
    V = 1 / (2 * N * (1 - sigma_sq))
    r = sigma_sq / (sigma_sq + V)
    log10BF = 0.5 * (np.log10(1 - r) + (r * z**2) / np.log(10))
    return log10BF

def coloc_abf(dataset1, dataset2,
              pval_col1='pval', pval_col2='pval',
              N1=None, N2=None):
    if N1 is None or N2 is None:
        if 'N' in dataset1.columns:
            N1 = dataset1['N'].iloc[0]
        if 'N' in dataset2.columns:
            N2 = dataset2['N'].iloc[0]
    if N1 is None or N2 is None:
        raise ValueError("必须提供两个数据集的样本量 N1 和 N2")

    merged = pd.merge(dataset1[[pval_col1]].reset_index(),
                      dataset2[[pval_col2]].reset_index(),
                      left_index=True, right_index=True, suffixes=('_1','_2'))
    if len(merged) == 0:
        print("警告：两个数据集没有共同的 SNP")
        return None

    bf1 = merged[pval_col1].apply(lambda p: approx_bf(p, N1))
    bf2 = merged[pval_col2].apply(lambda p: approx_bf(p, N2))

    pc1 = 1e-4
    pc2 = 1e-4
    p12 = 5e-5
    log_prior = {
        'H0': np.log(1 - pc1 - pc2 - p12),
        'H1': np.log(pc1 - p12),
        'H2': np.log(pc2 - p12),
        'H3': np.log(p12),
        'H4': np.log(p12)
    }
    log_bf1 = bf1 * np.log(10)
    log_bf2 = bf2 * np.log(10)

    log_post = pd.DataFrame({
        'H0': log_prior['H0'],
        'H1': log_prior['H1'] + log_bf1,
        'H2': log_prior['H2'] + log_bf2,
        'H3': log_prior['H3'] + log_bf1 + log_bf2,
        'H4': log_prior['H4'] + log_bf1 + log_bf2
    })
    log_sum = log_post.sum(axis=0)
    log_max = log_sum.max()
    log_sum_norm = log_sum - log_max
    pp = np.exp(log_sum_norm) / np.exp(log_sum_norm).sum()
    return dict(zip(['PP0','PP1','PP2','PP3','PP4'], pp))

# ======================== 3. SMR 分析（简化） ========================

def smr_analysis(eqtl_df, gwas_df, gene_name,
                 eqtl_beta_col='beta', eqtl_se_col='se',
                 gwas_beta_col='beta', gwas_se_col='se'):
    gene_eqtl = eqtl_df[eqtl_df['gene'] == gene_name].copy()
    if len(gene_eqtl) == 0:
        print(f"基因 {gene_name} 在 eQTL 数据中未找到。")
        return None
    top_snp = gene_eqtl.loc[gene_eqtl['pval'].idxmin()]
    snp_id = top_snp['SNP']
    beta_e = top_snp[eqtl_beta_col]
    se_e = top_snp[eqtl_se_col]

    gwas_snp = gwas_df[gwas_df['SNP'] == snp_id]
    if len(gwas_snp) == 0:
        print(f"SNP {snp_id} 在 GWAS 中未找到。")
        return None
    beta_g = gwas_snp.iloc[0][gwas_beta_col]
    se_g = gwas_snp.iloc[0][gwas_se_col]

    beta_smr = beta_g / beta_e
    se_smr = np.sqrt((se_g**2)/(beta_e**2) + (beta_g**2 * se_e**2)/(beta_e**4))
    pval = 2 * (1 - stats.norm.cdf(abs(beta_smr / se_smr)))

    return {
        'gene': gene_name, 'SNP': snp_id,
        'beta_e': beta_e, 'se_e': se_e,
        'beta_g': beta_g, 'se_g': se_g,
        'beta_smr': beta_smr, 'se_smr': se_smr, 'pval': pval
    }

# ======================== 4. 主程序 ========================

if __name__ == "__main__":
    files = {
        'LUSC': 'LUSC_tumor.trans_eQTL.txt',
        'LUAD': 'LUAD_tumor.trans_eQTL.txt',
        'survival': 'survival-eQTL_all_data.txt',
        'whole': 'whole_GWAS_eQTL_re_0711.txt'
    }

    # 第一步：预览文件（已在前次运行中完成，此处省略）
    # 请根据预览结果修改以下配置 =================================
    file_config = {
        'LUSC': {
            'sep': '\t',
            'has_header': True,
            'snp_col': 'SNP',
            'gene_col': 'gene',
            'beta_col': 'beta',
            'se_col': None,
            'tstat_col': 't-stat',
            'pval_col': 'p-value',
            'chr_col': None,
            'pos_col': None,
            'maf_col': None,
            'N': 500   # 请填入实际样本量（必须整数）
        },
        'LUAD': {
            'sep': '\t',
            'has_header': True,
            'snp_col': 'SNP',
            'gene_col': 'gene',
            'beta_col': 'beta',
            'se_col': None,
            'tstat_col': 't-stat',
            'pval_col': 'p-value',
            'chr_col': None,
            'pos_col': None,
            'maf_col': None,
            'N': 500
        },
        'survival': {
            'sep': '\t',
            'has_header': False,
            'snp_col': 1,           # 列索引 1 (rs号)
            'gene_col': None,
            'beta_col': 7,           # 假设列7是beta
            'se_col': 8,             # 假设列8是标准误
            'tstat_col': 9,          # 可选
            'pval_col': 5,           # 假设列5是p值
            'chr_col': 2,
            'pos_col': 3,
            'maf_col': None,
            'N': 500
        },
        'whole': {
            'sep': '\t',
            'has_header': True,
            'snp_col': 'SNP',
            'gene_col': 'gene',
            'beta_col': 'beta',
            'se_col': None,
            'tstat_col': 't-stat',
            'pval_col': 'p-value',
            'chr_col': 's_chr',
            'pos_col': 's_pos',
            'maf_col': None,
            'N': 500
        }
    }
    # ============================================================

    # 读取所有文件
    data = {}
    for name, cfg in file_config.items():
        print(f"\n正在读取 {name} 文件...")
        try:
            df = read_eqtl(
                files[name],
                snp_col=cfg['snp_col'],
                gene_col=cfg['gene_col'],
                beta_col=cfg['beta_col'],
                se_col=cfg['se_col'],
                pval_col=cfg['pval_col'],
                tstat_col=cfg.get('tstat_col'),
                maf_col=cfg.get('maf_col'),
                chr_col=cfg.get('chr_col'),
                pos_col=cfg.get('pos_col'),
                sep=cfg['sep'],
                has_header=cfg['has_header'],
                N=cfg.get('N')
            )
            data[name] = df
            print(f"成功读取 {len(df)} 行，列：{df.columns.tolist()}")
            print(df.head())
        except Exception as e:
            print(f"读取失败：{e}")
            print("请检查配置后重新运行。")
            exit(1)

    # 示例：提取 IL18 基因的 eQTL（仅对有 gene 列的文件）
    for name in ['LUSC', 'LUAD', 'whole']:
        if name in data and 'gene' in data[name].columns:
            il18 = extract_gene(data[name], 'IL18')
            print(f"\n{name} 中 IL18 的 eQTL 数量：{len(il18)}")
            if len(il18) > 0:
                print(il18[['SNP','beta','se','pval']].head())

    # 示例：区域提取（仅对 survival 和 whole 有效，因为它们有 chr/pos）
    if 'survival' in data and 'chr' in data['survival'].columns:
        # 提取某个区域，例如 chr2:125000000-125100000
        region = extract_region(data['survival'], '2', 125000000, 125100000)
        print(f"\nsurvival 中 chr2:125000000-125100000 区域 SNP 数量：{len(region)}")
        if len(region) > 0:
            print(region.head())

    print("\n所有文件读取完成。请根据您的分析需求进行后续操作。")