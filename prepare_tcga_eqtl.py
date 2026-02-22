#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取 TCGA LUAD/LUSC 肿瘤组织 cis-eQTL 中 NLRP3 通路基因
并转换为标准 MR 暴露格式
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ========== 配置 ==========
TARGET_GENES = ['NLRP3', 'CASP1', 'IL1B', 'IL18', 'PYCARD', 'IL18R1']
P_THRESHOLD = 1e-5  # 可根据需要调整，也可使用 FDR 列
DATA_DIR = Path("MR_Data")
OUTPUT_DIR = Path("TCGA_eQTL_MR")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================

def process_tcga_eqtl(file_path, cancer_type, eqtl_type):
    """
    处理单个 TCGA eQTL 文件
    列名示例：SNP	gene	beta	t-stat	p-value	FDR
    """
    print(f"\n处理文件: {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    print(f"  原始行数: {len(df)}")

    # 提取目标基因
    df_target = df[df['gene'].str.split('|').str[0].isin(TARGET_GENES)].copy()
    if len(df_target) == 0:
        print(f"  未找到目标基因")
        return None

    print(f"  目标基因 eQTL 数量: {len(df_target)}")

    # 根据 p-value 或 FDR 筛选显著 SNP
    if 'FDR' in df_target.columns:
        df_sig = df_target[df_target['FDR'] < 0.05].copy()
        print(f"  FDR < 0.05 筛选后: {len(df_sig)}")
    elif 'p-value' in df_target.columns:
        df_sig = df_target[df_target['p-value'] < P_THRESHOLD].copy()
        print(f"  p < {P_THRESHOLD} 筛选后: {len(df_sig)}")
    else:
        df_sig = df_target.copy()

    if len(df_sig) == 0:
        print(f"  无显著 eQTL，跳过")
        return None

    # 提取 SNP 信息（SNP 格式可能是 "rs199534:44824213:T:G"）
    # 我们保留完整 ID 作为 SNP 列
    df_sig['SNP'] = df_sig['SNP']
    df_sig['beta'] = df_sig['beta']
    df_sig['se'] = np.abs(df_sig['beta'] / df_sig['t-stat'])  # 从 t 统计量反推标准误
    df_sig['pval'] = df_sig['p-value'] if 'p-value' in df_sig.columns else df_sig['FDR']
    df_sig['effect_allele'] = None  # TCGA 数据通常不提供等位基因，留空
    df_sig['other_allele'] = None

    # 提取 rsID（例如从 "rs199534:44824213:T:G" 提取 "rs199534"）
    df_sig['SNP'] = df_sig['SNP'].str.split(':').str[0]
    print(f"  示例 SNP: {df_sig['SNP'].iloc[0] if len(df_sig) > 0 else '无'}")  # 调试用

    # 计算 F 统计量
    df_sig['f_stat'] = (df_sig['beta'] / df_sig['se']) ** 2

    # 保存为 MR 暴露格式
    output_cols = ['SNP', 'gene', 'beta', 'se', 'pval', 'effect_allele', 'other_allele', 'f_stat']
    output_file = OUTPUT_DIR / f"{cancer_type}_{eqtl_type}_target_eqtl.tsv"
    df_sig[output_cols].to_csv(output_file, sep='\t', index=False)
    print(f"  已保存: {output_file}")

    # 按基因汇总
    print(f"  各基因 SNP 数量:")
    print(df_sig['gene'].value_counts())

    return df_sig


def main():
    # 处理 LUAD cis-eQTL
    if Path("LUAD_tumor.cis_eQTL.txt").exists():
        process_tcga_eqtl("LUAD_tumor.cis_eQTL.txt", "LUAD", "cis")

    # 处理 LUSC cis-eQTL
    if Path("LUSC_tumor.cis_eQTL.txt").exists():
        process_tcga_eqtl("LUSC_tumor.cis_eQTL.txt", "LUSC", "cis")

    # 如需处理 trans-eQTL，可取消注释以下行
    # if Path("LUAD_tumor.trans_eQTL.txt").exists():
    #     process_tcga_eqtl("LUAD_tumor.trans_eQTL.txt", "LUAD", "trans")
    # if Path("LUSC_tumor.trans_eQTL.txt").exists():
    #     process_tcga_eqtl("LUSC_tumor.trans_eQTL.txt", "LUSC", "trans")


if __name__ == "__main__":
    main()