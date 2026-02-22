#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为共定位分析准备数据：为 TCGA eQTL 添加 chr, pos（从结局 GWAS 提取）
修复：处理结局文件中 SNP 重复的问题
"""

import pandas as pd
from pathlib import Path

# ========== 路径配置 ==========
WORK_DIR = Path("C:/Users/wswan/PycharmProjects/PythonProject2")
OUTCOME_FILE = WORK_DIR / "MR_Data/lung_cancer_outcome.tsv"
TCGA_DIR = WORK_DIR / "TCGA_eQTL_MR"
OUTPUT_DIR = WORK_DIR / "TCGA_MR_Results"
OUTPUT_DIR.mkdir(exist_ok=True)
# ==============================

# 1. 读取结局数据，建立 SNP → (chr, pos) 映射
print("读取结局数据...")
outcome = pd.read_csv(OUTCOME_FILE, sep='\t')

# 检查必需列
if 'chr' not in outcome.columns or 'pos' not in outcome.columns:
    print("错误：结局文件缺少 'chr' 或 'pos' 列，无法进行共定位。")
    print("当前列名：", list(outcome.columns))
    exit(1)

# 去除重复 SNP，保留第一行
outcome_unique = outcome.drop_duplicates(subset='SNP', keep='first')
print(f"结局原始 SNP 数: {len(outcome)}，去重后 SNP 数: {len(outcome_unique)}")

# 构建映射字典：SNP -> {'chr': chr, 'pos': pos}
snp_pos = outcome_unique.set_index('SNP')[['chr', 'pos']].to_dict('index')
print(f"映射构建完成，包含 {len(snp_pos)} 个 SNP")

# 2. 处理 LUAD IL18 eQTL
eqtl_file = TCGA_DIR / "LUAD_cis_target_eqtl.tsv"
if eqtl_file.exists():
    print("\n处理 LUAD IL18 eQTL...")
    eqtl = pd.read_csv(eqtl_file, sep='\t')
    eqtl['gene'] = eqtl['gene'].str.split('|').str[0]
    eqtl = eqtl[eqtl['gene'] == 'IL18'].copy()

    # 添加 chr, pos
    eqtl['chr'] = eqtl['SNP'].map(lambda x: snp_pos.get(x, {}).get('chr', None))
    eqtl['pos'] = eqtl['SNP'].map(lambda x: snp_pos.get(x, {}).get('pos', None))

    # 删除位置缺失的 SNP
    eqtl = eqtl.dropna(subset=['chr', 'pos'])
    eqtl['chr'] = eqtl['chr'].astype(int)
    eqtl['pos'] = eqtl['pos'].astype(int)

    output_file = OUTPUT_DIR / "LUAD_IL18_coloc_input.tsv"
    eqtl.to_csv(output_file, sep='\t', index=False)
    print(f"  LUAD IL18: {len(eqtl)} SNPs with positions saved.")
else:
    print("LUAD eQTL 文件不存在，跳过")

# 3. 处理 LUSC IL18 eQTL
eqtl_file = TCGA_DIR / "LUSC_cis_target_eqtl.tsv"
if eqtl_file.exists():
    print("\n处理 LUSC IL18 eQTL...")
    eqtl = pd.read_csv(eqtl_file, sep='\t')
    eqtl['gene'] = eqtl['gene'].str.split('|').str[0]
    eqtl = eqtl[eqtl['gene'] == 'IL18'].copy()

    eqtl['chr'] = eqtl['SNP'].map(lambda x: snp_pos.get(x, {}).get('chr', None))
    eqtl['pos'] = eqtl['SNP'].map(lambda x: snp_pos.get(x, {}).get('pos', None))

    eqtl = eqtl.dropna(subset=['chr', 'pos'])
    eqtl['chr'] = eqtl['chr'].astype(int)
    eqtl['pos'] = eqtl['pos'].astype(int)

    output_file = OUTPUT_DIR / "LUSC_IL18_coloc_input.tsv"
    eqtl.to_csv(output_file, sep='\t', index=False)
    print(f"  LUSC IL18: {len(eqtl)} SNPs with positions saved.")
else:
    print("LUSC eQTL 文件不存在，跳过")

print("\n共定位输入文件准备完成！")