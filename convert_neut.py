#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 neut.tsv.gz（VCF 格式）转换为标准 MR 中介数据
复用 prepare_mr_data.py 中的 convert_vcf_to_mr 逻辑
"""

import pandas as pd
import gzip
from pathlib import Path

WORK_DIR = Path("C:/Users/wswan/PycharmProjects/PythonProject2")
NEUT_FILE = WORK_DIR / "neut.tsv.gz"
OUTPUT_FILE = WORK_DIR / "MR_Data/neutrophil_mediator.tsv"


def convert_vcf_to_mr(vcf_file, output_file):
    """VCF → MR 格式转换（专用于 neut.tsv.gz）"""
    data = []
    snp_count = 0

    with gzip.open(vcf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 10:
                continue

            # 基本列
            chrom = fields[0]
            pos = fields[1]
            rsid = fields[2] if fields[2] != '.' else f"{chrom}:{pos}"
            ref = fields[3]
            alt = fields[4]
            format_col = fields[8]
            sample_col = fields[9]

            # 解析样本数据
            format_keys = format_col.split(':')
            sample_values = sample_col.split(':')
            if len(format_keys) == len(sample_values):
                sample_dict = dict(zip(format_keys, sample_values))
                es = sample_dict.get('ES')
                se = sample_dict.get('SE')
                lp = sample_dict.get('LP')
                if es and se and lp:
                    try:
                        beta = float(es)
                        se_val = float(se)
                        lp_val = float(lp)
                        pval = 10 ** (-lp_val)
                        data.append({
                            'SNP': rsid,
                            'beta': beta,
                            'se': se_val,
                            'pval': pval
                        })
                        snp_count += 1
                    except:
                        continue
            if snp_count % 100000 == 0 and snp_count > 0:
                print(f"  已处理 {snp_count:,} 个SNP...")

    df = pd.DataFrame(data)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"转换完成！共 {len(df)} 个SNP，已保存至 {output_file}")
    return df


if __name__ == "__main__":
    if not NEUT_FILE.exists():
        print(f"错误：文件 {NEUT_FILE} 不存在")
    else:
        convert_vcf_to_mr(NEUT_FILE, OUTPUT_FILE)