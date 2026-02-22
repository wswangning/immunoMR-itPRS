"""
eQTL 数据处理与共定位分析模块（完整版）
包含：
- 读取四个 eQTL 文件
- 与 GWAS 数据整合
- 共定位分析（近似贝叶斯）
- SMR 分析
- 区域可视化
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
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
        keep_cols.extend(['chr', 'pos'])

    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df.dropna(subset=['beta', 'se', 'pval'], inplace=True)
    df['pval'] = df['pval'].clip(lower=1e-300)
    if N is not None:
        df['N'] = N
    return df


def extract_gene(df, gene_name):
    if 'gene' not in df.columns:
        raise ValueError("DataFrame 缺少 'gene' 列")
    return df[df['gene'].astype(str).str.upper() == gene_name.upper()].copy()


def extract_region(df, chrom, start, end):
    required = ['chr', 'pos']
    if not all(col in df.columns for col in required):
        raise ValueError("需要 'chr' 和 'pos' 列进行区域提取")
    chrom = str(chrom).replace('chr', '')
    df['chr'] = df['chr'].astype(str).str.replace('chr', '')
    mask = (df['chr'] == chrom) & (df['pos'].between(start, end))
    return df[mask].copy()


# ======================== 2. 共定位分析（近似贝叶斯）==================

def approx_bf(pval, N, sigma_sq=0.15):
    """Wakefield 近似贝叶斯因子计算
    参数：
        pval: P 值
        N: 样本量
        sigma_sq: 先验方差（默认 0.15^2）
    返回：
        log10(BF)
    """
    # 使用 isf (inverse survival function) 计算上侧分位数，等价于 q = 1 - pval/2
    z = stats.norm.isf(pval / 2)   # 对于双尾检验，上侧分位数对应 pval/2
    V = 1 / (2 * N * (1 - sigma_sq))
    r = sigma_sq / (sigma_sq + V)
    log10BF = 0.5 * (np.log10(1 - r) + (r * z**2) / np.log(10))
    return log10BF



# 在 approx_bf 后面添加：
def coloc_abf_simple(pval1, pval2, N1, N2):
    """简化版共定位，直接传入对齐的P值向量"""
    bf1 = [approx_bf(p, N1) for p in pval1]
    bf2 = [approx_bf(p, N2) for p in pval2]
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
    log_sum_norm = log_sum - log_max
    pp = np.exp(log_sum_norm) / np.exp(log_sum_norm).sum()
    return dict(zip(['PP0', 'PP1', 'PP2', 'PP3', 'PP4'], pp))


# 简化版共定位函数（直接传入对齐的P值向量）
def coloc_abf_simple(pval1, pval2, N1, N2):
    """pval1, pval2: 两个P值数组（长度相同，已按SNP对齐）"""
    bf1 = [approx_bf(p, N1) for p in pval1]
    bf2 = [approx_bf(p, N2) for p in pval2]
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
    log_sum_norm = log_sum - log_max
    pp = np.exp(log_sum_norm) / np.exp(log_sum_norm).sum()
    return dict(zip(['PP0', 'PP1', 'PP2', 'PP3', 'PP4'], pp))

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
    return dict(zip(['PP0', 'PP1', 'PP2', 'PP3', 'PP4'], pp))


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
    se_smr = np.sqrt((se_g ** 2) / (beta_e ** 2) + (beta_g ** 2 * se_e ** 2) / (beta_e ** 4))
    pval = 2 * (1 - stats.norm.cdf(abs(beta_smr / se_smr)))

    return {
        'gene': gene_name, 'SNP': snp_id,
        'beta_e': beta_e, 'se_e': se_e,
        'beta_g': beta_g, 'se_g': se_g,
        'beta_smr': beta_smr, 'se_smr': se_smr, 'pval': pval
    }


# ======================== 4. 区域可视化 ========================

def plot_region(eqtl_df, gwas_df, chrom, start, end, gene_name=None):
    """绘制指定区域eQTL和GWAS的P值分布"""
    eqtl_region = extract_region(eqtl_df, chrom, start, end)
    gwas_region = gwas_df[gwas_df['SNP'].isin(eqtl_region['SNP'])]
    merged = pd.merge(eqtl_region, gwas_region, on='SNP', suffixes=('_eqtl', '_gwas'))

    if len(merged) == 0:
        print("无重叠SNP，无法绘图")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # eQTL
    ax1.scatter(merged['pos'], -np.log10(merged['pval_eqtl']), c='blue', alpha=0.6)
    ax1.set_ylabel('-log10(P) eQTL')
    ax1.set_title(f'{chrom}:{start}-{end}' + (f' for {gene_name}' if gene_name else ''))
    ax1.axhline(-np.log10(5e-8), color='red', linestyle='--', label='genome-wide sig')
    ax1.legend()
    # GWAS
    ax2.scatter(merged['pos'], -np.log10(merged['pval_gwas']), c='green', alpha=0.6)
    ax2.set_xlabel('Position (bp)')
    ax2.set_ylabel('-log10(P) GWAS')
    ax2.axhline(-np.log10(5e-8), color='red', linestyle='--')
    plt.tight_layout()
    plt.show()


# ======================== 5. 主程序 ========================

if __name__ == "__main__":
    # 文件路径（请确保文件在当前目录或修改为绝对路径）
    files = {
        'LUSC': 'LUSC_tumor.trans_eQTL.txt',
        'LUAD': 'LUAD_tumor.trans_eQTL.txt',
        'survival': 'survival-eQTL_all_data.txt',
        'whole': 'whole_GWAS_eQTL_re_0711.txt'
    }

    # ========== 根据实际文件结构修改以下配置 ==========
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
            'N': 500  # 请根据实际样本量修改
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
            'snp_col': 1,  # 列索引1 (rs号)
            'gene_col': None,
            'beta_col': 7,  # 假设列7是beta
            'se_col': 8,  # 假设列8是标准误
            'tstat_col': 9,  # 可选
            'pval_col': 5,  # 假设列5是p值
            'chr_col': 2,
            'pos_col': 3,
            'maf_col': None,
            'N': 500  # 样本量，请修改
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
            'N': 500  # 样本量，请修改
        }
    }
    # ==================================================

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

    # ========== 加载 GWAS 数据 ==========
    # 如果使用真实GWAS，请取消下面一行的注释并指定文件路径
    # gwas_df = pd.read_csv('your_lung_cancer_gwas.txt', sep='\t')

    # 如果暂时没有真实GWAS，使用以下模拟代码（包含目标基因SNP，仅用于测试）
    print("\n正在生成包含目标基因SNP的模拟GWAS数据（用于测试流程）...")
    np.random.seed(123)
    # 获取IL18的所有SNP
    il18_snps = data['whole'][data['whole']['gene'] == 'IL18']['SNP'].tolist()
    # 从whole中随机选择其他SNP
    all_snps = data['whole']['SNP'].unique()
    other_snps = np.setdiff1d(all_snps, il18_snps)
    selected_other = np.random.choice(other_snps, size=min(500, len(other_snps)), replace=False)
    gwas_snps = il18_snps + selected_other.tolist()
    np.random.shuffle(gwas_snps)

    gwas_df = pd.DataFrame({
        'SNP': gwas_snps,
        'beta': np.random.normal(0, 0.1, len(gwas_snps)),
        'se': np.random.uniform(0.02, 0.05, len(gwas_snps)),
        'pval': np.random.uniform(0, 1, len(gwas_snps)),
        'A1': np.random.choice(['A', 'C', 'G', 'T'], len(gwas_snps)),
        'A2': np.random.choice(['A', 'C', 'G', 'T'], len(gwas_snps))
    })
    # 将IL18的SNP的P值设得更显著
    il18_mask = gwas_df['SNP'].isin(il18_snps)
    gwas_df.loc[il18_mask, 'pval'] = np.random.uniform(1e-10, 1e-6, il18_mask.sum())
    print(f"模拟GWAS包含 {len(il18_snps)} 个IL18的SNP，总SNP数 {len(gwas_df)}")
    # 实际使用时，请注释掉以上模拟部分，并取消下面一行的注释：
    # gwas_df = pd.read_csv('your_gwas_file.txt', sep='\t')

    # ========== 6. 共定位分析（以IL18为例）==========
    target_gene = 'IL18'
    if target_gene in data['whole']['gene'].values:
        eqtl_gene = extract_gene(data['whole'], target_gene)
        print(f"\n基因 {target_gene} 在 whole 数据中有 {len(eqtl_gene)} 个eQTL")

        merged = pd.merge(eqtl_gene, gwas_df, on='SNP', suffixes=('_eqtl', '_gwas'))
        if len(merged) == 0:
            print("警告：没有与GWAS共同的SNP")
        else:
            print(f"与GWAS共同SNP数量：{len(merged)}")
            N_eqtl = merged['N'].iloc[0]  # whole的样本量
            N_gwas = 50000  # 请根据实际GWAS样本量修改
            coloc_res = coloc_abf_simple(merged['pval_eqtl'], merged['pval_gwas'], N_eqtl, N_gwas)
            if coloc_res:
                print(f"\n{target_gene} 共定位结果：")
                for k, v in coloc_res.items():
                    print(f"  {k}: {v:.4f}")
    else:
        print(f"基因 {target_gene} 在 whole 数据中不存在")

    # ========== 7. 对survival中的特定区域进行共定位 ==========
    # 例如：chr2:125075680附近区域
    chrom = '2'
    pos = 125075680
    window = 500000
    start = pos - window
    end = pos + window
    if 'survival' in data and 'chr' in data['survival'].columns:
        region_survival = extract_region(data['survival'], chrom, start, end)
        print(f"\nsurvival 中 {chrom}:{start}-{end} 区域有 {len(region_survival)} 个SNP")
        merged_region = pd.merge(region_survival, gwas_df, on='SNP', suffixes=('_surv', '_gwas'))
        if len(merged_region) > 0:
            N_surv = merged_region['N'].iloc[0]
            N_gwas = 50000
            coloc_region = coloc_abf_simple(merged_region['pval_surv'], merged_region['pval_gwas'], N_surv, N_gwas)
            if coloc_region:
                print(f"\n区域 {chrom}:{start}-{end} 共定位结果：")
                for k, v in coloc_region.items():
                    print(f"  {k}: {v:.4f}")

    # ========== 8. SMR分析（以IL18为例）==========
    if target_gene in data['whole']['gene'].values:
        smr_res = smr_analysis(data['whole'], gwas_df, target_gene)
        if smr_res:
            print(f"\nSMR分析结果 for {target_gene}:")
            print(f"  最显著SNP: {smr_res['SNP']}")
            print(f"  eQTL效应: beta={smr_res['beta_e']:.3f}, se={smr_res['se_e']:.3f}")
            print(f"  GWAS效应: beta={smr_res['beta_g']:.3f}, se={smr_res['se_g']:.3f}")
            print(f"  SMR估计: beta={smr_res['beta_smr']:.3f}, se={smr_res['se_smr']:.3f}, p={smr_res['pval']:.3e}")

    # ========== 9. 批量分析多个免疫相关基因 ==========
    immune_genes = ['IL18', 'IL1B', 'TNF', 'IFNG', 'NLRP3', 'NFE2L2', 'GPX4', 'SLC7A11', 'ACSL4']
    print("\n批量SMR分析结果：")
    for gene in immune_genes:
        if gene in data['whole']['gene'].values:
            res = smr_analysis(data['whole'], gwas_df, gene)
            if res:
                print(f"{gene}: beta={res['beta_smr']:.3f}, p={res['pval']:.3e}")
            else:
                print(f"{gene}: 分析失败")
        else:
            print(f"{gene}: 不存在于whole数据中")

    # ========== 10. 区域可视化（以IL18为例）==========
    if target_gene in data['whole']['gene'].values:
        il18_snps = extract_gene(data['whole'], target_gene)
        if len(il18_snps) > 0:
            chrom_il18 = il18_snps['chr'].iloc[0]
            pos_min = il18_snps['pos'].min()
            pos_max = il18_snps['pos'].max()
            start_plot = pos_min - 500000
            end_plot = pos_max + 500000
            plot_region(data['whole'], gwas_df, chrom_il18, start_plot, end_plot, gene_name=target_gene)

    print("\n分析完成。")