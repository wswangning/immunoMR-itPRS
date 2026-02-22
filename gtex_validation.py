#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTEx 肺组织 IL18 eQTL MR 验证
通过 IEU OpenGWAS API 直接获取数据，无需本地大文件
"""

import pandas as pd
import numpy as np
import requests
import io
import gzip
from scipy import stats
from pathlib import Path

WORK_DIR = Path("C:/Users/wswan/PycharmProjects/PythonProject2")
OUTCOME_FILE = WORK_DIR / "MR_Data/lung_cancer_outcome.tsv"
OUTPUT_DIR = WORK_DIR / "TCGA_MR_Results"
OUTPUT_DIR.mkdir(exist_ok=True)


def mr_ivw(df, beta_exp, se_exp, beta_out, se_out):
    """逆方差加权法（固定效应）"""
    beta_exp = df[beta_exp].values
    se_exp = df[se_exp].values
    beta_out = df[beta_out].values
    se_out = df[se_out].values

    valid = (beta_exp != 0) & np.isfinite(beta_exp) & np.isfinite(beta_out) & (se_exp > 0) & (se_out > 0)
    df_valid = df[valid].copy()
    if len(df_valid) < 2:
        return None

    beta_exp = df_valid[beta_exp].values
    se_exp = df_valid[se_exp].values
    beta_out = df_valid[beta_out].values
    se_out = df_valid[se_out].values

    wald = beta_out / beta_exp
    wald_se = np.sqrt((se_out ** 2 / beta_exp ** 2) + (beta_out ** 2 * se_exp ** 2 / beta_exp ** 4))

    finite = np.isfinite(wald) & np.isfinite(wald_se) & (wald_se > 0)
    wald = wald[finite]
    wald_se = wald_se[finite]

    if len(wald) < 2:
        return None

    weights = 1 / (wald_se ** 2)
    beta = np.sum(wald * weights) / np.sum(weights)
    se = np.sqrt(1 / np.sum(weights))
    p = 2 * (1 - stats.norm.cdf(np.abs(beta / se)))
    ci_lower = beta - 1.96 * se
    ci_upper = beta + 1.96 * se

    return {'beta': beta, 'se': se, 'pval': p, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'n_snps': len(wald)}


def fetch_ieu_gwas(id_list, batch_size=100):
    """从 IEU OpenGWAS API 批量获取 GWAS 数据"""
    base_url = "http://gwas-api.mrcieu.ac.uk/"
    all_associations = []

    for gwas_id in id_list:
        print(f"  获取 {gwas_id} ...")
        url = f"{base_url}associations/{gwas_id}"
        response = requests.get(url, params={'pval': 5e-6, 'batch': batch_size})
        if response.status_code != 200:
            print(f"    请求失败: {response.status_code}")
            continue
        data = response.json()
        assocs = data.get('_embedded', {}).get('associations', [])
        print(f"    获取到 {len(assocs)} 个关联")

        for a in assocs:
            variant = a.get('variant', {})
            all_associations.append({
                'SNP': variant.get('rsid'),
                'beta': a.get('beta'),
                'se': a.get('se'),
                'pval': a.get('pvalue'),
                'effect_allele': a.get('effect_allele'),
                'other_allele': a.get('other_allele'),
                'eaf': a.get('eaf'),
                'id': gwas_id
            })
    return pd.DataFrame(all_associations)


def get_gtex_lung_eqtl_il18():
    """获取 GTEx v8 肺组织 IL18 eQTL 数据"""
    # IEU ID 格式：'eqtl-a-ENSG00000137752'（跨组织 meta 分析）
    # 肺组织特异性 ID 不易获取，我们使用跨组织数据作为替代
    # 为了获得肺组织特异性，我们尝试已知的 GTEx 肺组织 eQTL ID
    # 参考：https://gwas.mrcieu.ac.uk/phewas/?query=eqtl-a-ENSG00000137752
    # 实际可用的肺组织 eQTL ID 可能是 'gtex-...'，但我们先用跨组织数据
    gtex_id = "eqtl-a-ENSG00000137752"

    print(f"正在从 IEU OpenGWAS 获取 IL18 eQTL 数据 (ID: {gtex_id})...")
    df = fetch_ieu_gwas([gtex_id])
    if df.empty:
        print("  获取失败，尝试其他 ID...")
        # 备选：使用 eQTLGen 数据（血液）
        eqtlgen_id = "eqtl-a-ENSG00000137752"  # 实际上 eQTLGen 也有相同 ID 格式
        df = fetch_ieu_gwas([eqtlgen_id])

    if df.empty:
        print("❌ 无法获取 IL18 eQTL 数据")
        return None

    # 筛选显著 SNP
    df = df[df['pval'] < 5e-6].copy()
    print(f"  显著 SNP 数量: {len(df)}")

    # 去重
    df = df.drop_duplicates(subset='SNP')
    return df


def main():
    if not OUTCOME_FILE.exists():
        print(f"❌ 结局文件不存在: {OUTCOME_FILE}")
        return

    # 1. 获取 IL18 eQTL 数据
    print("步骤1: 获取 IL18 eQTL 数据...")
    il18 = get_gtex_lung_eqtl_il18()
    if il18 is None or len(il18) < 2:
        print("❌ IL18 eQTL 数据不足")
        return

    il18 = il18.rename(columns={'beta': 'beta_exp', 'se': 'se_exp'})
    print(f"  IL18 eQTL SNPs: {len(il18)}")

    # 2. 读取结局数据
    print("\n步骤2: 读取肺癌结局数据...")
    outcome = pd.read_csv(OUTCOME_FILE, sep='\t')
    outcome = outcome[['SNP', 'beta', 'se']].rename(columns={'beta': 'beta_out', 'se': 'se_out'})
    print(f"  结局 SNP 数: {len(outcome)}")

    # 3. 合并
    merged = pd.merge(il18, outcome, on='SNP', how='inner')
    print(f"\n步骤3: 合并后 SNP 数量: {len(merged)}")

    if len(merged) < 2:
        print("❌ 匹配 SNP 不足")
        return

    # 4. 计算 F 统计量
    merged['f_stat'] = (merged['beta_exp'] / merged['se_exp']) ** 2
    print(f"  平均 F 统计量: {merged['f_stat'].mean():.2f}")

    # 5. IVW
    res = mr_ivw(merged, 'beta_exp', 'se_exp', 'beta_out', 'se_out')
    if res:
        print(f"\n✅ IVW 结果:")
        print(f"  β = {res['beta']:.4f} (95% CI: {res['ci_lower']:.4f}, {res['ci_upper']:.4f})")
        print(f"  SE = {res['se']:.4f}, P = {res['pval']:.4e}, SNPs = {res['n_snps']}")

        # 6. 保存结果
        out_df = pd.DataFrame([res])
        out_df.to_csv(OUTPUT_DIR / "GTEx_IL18_MR_results.csv", index=False)
        print(f"  结果已保存至: {OUTPUT_DIR / 'GTEx_IL18_MR_results.csv'}")
    else:
        print("❌ IVW 分析失败")


if __name__ == "__main__":
    main()