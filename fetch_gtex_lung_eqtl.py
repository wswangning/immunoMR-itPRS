#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTEx v8 肺组织 eQTL 下载器（自动/手动双模式）
- 自动尝试下载，若失败则给出精确的手动下载指令
"""

import pandas as pd
import requests
import shutil
from pathlib import Path

OUTPUT_DIR = Path("MR_Data")
OUTPUT_DIR.mkdir(exist_ok=True)

# GTEx v8 肺组织显著 eQTL 官方链接（2026年仍有效）
URL = "https://storage.googleapis.com/gtex_analysis_v8/single_tissue_qtl_data/GTEx_Analysis_v8_eQTL/Lung.v8.signif_variant_gene_pairs.txt.gz"
LOCAL_GZ = OUTPUT_DIR / "Lung_eQTL.txt.gz"

print("正在尝试自动下载 GTEx v8 肺组织 eQTL...")
print(f"URL: {URL}")

# 模拟浏览器请求头，避免被重定向
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    with requests.get(URL, stream=True, headers=headers, timeout=30) as r:
        r.raise_for_status()
        # 检查内容类型是否为 gzip
        content_type = r.headers.get('Content-Type', '')
        if 'gzip' not in content_type and 'octet-stream' not in content_type:
            print("警告：返回的内容类型不是 gzip，可能被重定向。")
            print(f"Content-Type: {content_type}")
            # 尝试读取前几个字节检查是否为 gzip 魔数
            first_bytes = r.raw.read(2)
            if first_bytes != b'\x1f\x8b':
                raise ValueError("下载的文件不是有效的 gzip 格式")
            r.raw.seek(0)  # 重置流位置
        with open(LOCAL_GZ, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    print(f"✅ 自动下载成功！文件已保存至: {LOCAL_GZ}")
except Exception as e:
    print(f"❌ 自动下载失败: {e}")
    print("\n" + "=" * 60)
    print("请按照以下步骤手动下载：")
    print("1. 访问 GTEx 门户网站：https://gtexportal.org/home/datasets")
    print("2. 点击 'GTEx Analysis V8' -> 'eQTL Summary Statistics'")
    print("3. 找到 'Lung' 组织，下载文件：")
    print("   Lung.v8.signif_variant_gene_pairs.txt.gz")
    print("4. 将下载的文件放置于：", LOCAL_GZ)
    print("5. 重新运行此脚本（会自动跳过下载，直接处理）")
    print("=" * 60)
    # 如果文件已存在（用户手动放置），继续执行
    if not LOCAL_GZ.exists():
        exit(1)

# 如果文件已存在（自动下载成功或用户手动放置），进行后续处理
if LOCAL_GZ.exists():
    print("\n开始处理 eQTL 数据...")

    # 读取数据（仅读取需要的列以节省内存）
    cols = ['variant_id', 'gene_name', 'slope', 'slope_se', 'pval_nominal']
    # 分块读取，避免内存爆炸
    chunk_iter = pd.read_csv(LOCAL_GZ, sep='\t', usecols=cols,
                             compression='gzip', chunksize=100000)

    target_genes = ['NLRP3', 'CASP1', 'IL1B', 'IL18']
    target_dfs = []
    for i, chunk in enumerate(chunk_iter):
        chunk_target = chunk[chunk['gene_name'].isin(target_genes)]
        if len(chunk_target) > 0:
            target_dfs.append(chunk_target)
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1} 个数据块...")

    if target_dfs:
        df_target = pd.concat(target_dfs, ignore_index=True)
        print(f"目标基因 eQTL 数量: {len(df_target)}")

        # 提取 SNP 标识（去除 _b38 后缀）
        df_target['SNP'] = df_target['variant_id'].str.replace('_b38$', '', regex=True)
        df_target['beta'] = df_target['slope']
        df_target['se'] = df_target['slope_se']
        df_target['pval'] = df_target['pval_nominal']
        df_target['effect_allele'] = None
        df_target['other_allele'] = None

        # 保存为 MR 标准格式
        output_file = OUTPUT_DIR / "GTEx_lung_eqtl_target.tsv"
        df_target[['SNP', 'beta', 'se', 'pval', 'effect_allele', 'other_allele', 'gene_name']].to_csv(
            output_file, sep='\t', index=False)
        print(f"✅ 已保存至: {output_file}")
    else:
        print("❌ 未找到目标基因的 eQTL，请检查基因名是否拼写正确。")