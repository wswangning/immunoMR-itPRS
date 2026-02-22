#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
免疫毒性亚型聚类分析（最终版）
修复：使用 effect_df.corr() 计算列间相关，避免内存溢出。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_DIR = "./MR_Data/"
OUTPUT_DIR = "./ML_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHENO_FILES = {
    'lymphocyte': 'lymphocyte_mediator.tsv',
    'monocyte': 'monocyte_mediator.tsv',
    'IL1B': 'IL1b_exposure.tsv',
    'IL18': 'IL18_exposure.tsv',
    'TNFa': 'TNFa_exposure.tsv',
    'inflammation': 'inflammation_mediator.tsv',
}

# ==================== 工具函数 ====================
def load_gwas_file(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, sep='\t', low_memory=False)
    except:
        try:
            df = pd.read_csv(filepath, sep=',', low_memory=False)
        except:
            return None
    rename_dict = {}
    if 'SNP' not in df.columns:
        for col in df.columns:
            if col.upper() in ['SNP', 'RSID', 'RS', 'VARIANT', 'MARKERNAME']:
                rename_dict[col] = 'SNP'
                break
    if 'beta' not in df.columns:
        for col in df.columns:
            if col.upper() in ['BETA', 'EFFECT', 'OR', 'B']:
                rename_dict[col] = 'beta'
                break
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    if 'SNP' not in df.columns or 'beta' not in df.columns:
        return None
    df = df[['SNP', 'beta']].dropna()
    df = df.drop_duplicates(subset=['SNP'])
    return df

def build_effect_matrix(pheno_dfs):
    all_snps = set()
    for df in pheno_dfs.values():
        all_snps.update(df['SNP'])
    all_snps = sorted(all_snps)
    effect_df = pd.DataFrame({'SNP': all_snps})
    for pheno, df in pheno_dfs.items():
        sub = df[['SNP', 'beta']].rename(columns={'beta': pheno})
        effect_df = effect_df.merge(sub, on='SNP', how='left')
    effect_df = effect_df.set_index('SNP').fillna(0)
    return effect_df

def compute_genetic_correlation(effect_df):
    """基于SNP效应向量计算表型间的遗传相关（皮尔逊相关）"""
    return effect_df.corr()  # 计算列间相关

def plot_consensus_cdf(consensus_matrices):
    plt.figure(figsize=(8,5))
    for k, cmat in consensus_matrices.items():
        vals = cmat[np.triu_indices_from(cmat, k=1)].flatten()
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
        plt.plot(sorted_vals, cdf, label=f'k={k}')
    plt.xlabel('Consensus index')
    plt.ylabel('CDF')
    plt.legend()
    plt.title('Consensus matrix CDF')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'consensus_cdf.png'), dpi=150)
    plt.close()

# ==================== 主流程 ====================
def main():
    print("="*60)
    print("免疫毒性亚型聚类分析（最终版）")
    print("="*60)

    # 加载数据
    pheno_dfs = {}
    for name, fname in PHENO_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        df = load_gwas_file(path)
        if df is not None:
            pheno_dfs[name] = df
            print(f"加载表型 {name}: {len(df)} 个SNP")
        else:
            print(f"表型 {name} 文件缺失，跳过")

    if len(pheno_dfs) < 3:
        print("错误：有效表型不足3个")
        return

    effect_mat = build_effect_matrix(pheno_dfs)
    print(f"效应矩阵形状：{effect_mat.shape}")
    effect_mat.to_csv(os.path.join(OUTPUT_DIR, 'effect_matrix.csv'))

    corr = compute_genetic_correlation(effect_mat)
    corr.to_csv(os.path.join(OUTPUT_DIR, 'genetic_correlation.csv'))
    print("遗传相关矩阵已保存")

    # 共识聚类
    n_traits = effect_mat.shape[1]
    n_clusters_range = range(2, min(6, n_traits+1))
    n_iter = 50
    consensus = {}
    sil_scores = []
    print("\n开始共识聚类...")
    for k in n_clusters_range:
        print(f"  k={k}...")
        cmat = np.zeros((n_traits, n_traits))
        sil_vals = []
        for seed in range(n_iter):
            sample = effect_mat.sample(frac=0.8, axis=0, replace=True, random_state=seed)
            X = sample.T.values
            X = StandardScaler().fit_transform(X)
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labs = km.fit_predict(X)
            for i in range(n_traits):
                for j in range(n_traits):
                    if labs[i] == labs[j]:
                        cmat[i,j] += 1
            if len(np.unique(labs)) > 1:
                sil_vals.append(silhouette_score(X, labs))
        cmat /= n_iter
        consensus[k] = cmat
        sil_scores.append(np.mean(sil_vals) if sil_vals else -1)

    # 选择最佳k
    best_k = list(consensus.keys())[np.argmax(sil_scores)]
    print(f"最佳聚类数：{best_k}，平均轮廓系数={max(sil_scores):.3f}")
    plot_consensus_cdf(consensus)

    # 用最佳k重新聚类所有SNP（完整数据）
    X_full = effect_mat.T.values
    X_full_scaled = StandardScaler().fit_transform(X_full)
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(X_full_scaled)
    cluster_df = pd.DataFrame({'trait': effect_mat.columns, 'cluster': final_labels})
    cluster_df.to_csv(os.path.join(OUTPUT_DIR, 'immune_subtypes.csv'), index=False)
    print("聚类标签已保存，分布：")
    print(cluster_df['cluster'].value_counts().sort_index())

    # 绘制热图
    plt.figure(figsize=(8,6))
    sns.clustermap(corr, method='ward', cmap='RdBu_r', center=0,
                   figsize=(8,6), dendrogram_ratio=0.2,
                   cbar_pos=(0.02, 0.8, 0.03, 0.15))
    plt.title('Genetic correlation of immune traits')
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_clustermap.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=corr.columns, yticklabels=corr.columns)
    plt.title('Genetic correlation matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()

    print("\n所有分析完成！结果保存在：", OUTPUT_DIR)

if __name__ == '__main__':
    main()