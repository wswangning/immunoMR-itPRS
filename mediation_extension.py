#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中介分析扩展：IL18 eQTL → 单核细胞 / 中性粒细胞 → 肺癌
完全自包含，无需依赖其他文件
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ========== 配置 ==========
WORK_DIR = Path("C:/Users/wswan/PycharmProjects/PythonProject2")
OUTCOME_FILE = WORK_DIR / "MR_Data/lung_cancer_outcome.tsv"
TCGA_DIR = WORK_DIR / "TCGA_eQTL_MR"
MEDIATOR_FILES = {
    'monocyte': WORK_DIR / "MR_Data/monocyte_mediator.tsv",
    'neutrophil': WORK_DIR / "MR_Data/neutrophil_mediator.tsv"
}
OUTPUT_DIR = WORK_DIR / "TCGA_MR_Results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================

def mediation_analysis_improved(exposure_df, mediator_df, outcome_df, n_bootstrap=1000):
    """
    两步法中介分析（IVW + Bootstrap CI）
    完全独立函数，可直接使用
    """
    # 第一步：暴露 → 中介
    exp_med = pd.merge(exposure_df, mediator_df, on='SNP', suffixes=('_exp', '_med'), how='inner')
    exp_med = exp_med[(exp_med['beta_exp'] != 0) & (exp_med['beta_med'] != 0)]
    if len(exp_med) < 2:
        return None
    weights_a = 1 / (exp_med['se_med'] ** 2 / exp_med['beta_exp'] ** 2 +
                     exp_med['beta_med'] ** 2 * exp_med['se_exp'] ** 2 / exp_med['beta_exp'] ** 4)
    a = np.average(exp_med['beta_med'] / exp_med['beta_exp'], weights=weights_a)
    se_a = np.sqrt(1 / np.sum(weights_a))
    p_a = 2 * (1 - stats.norm.cdf(np.abs(a / se_a))) if se_a > 0 else 1.0

    # 第二步：中介 → 结局
    med_out = pd.merge(mediator_df, outcome_df, on='SNP', suffixes=('_med', '_out'), how='inner')
    med_out = med_out[(med_out['beta_med'] != 0) & (med_out['beta_out'] != 0)]
    if len(med_out) < 2:
        return None
    weights_b = 1 / (med_out['se_out'] ** 2 / med_out['beta_med'] ** 2 +
                     med_out['beta_out'] ** 2 * med_out['se_med'] ** 2 / med_out['beta_med'] ** 4)
    b = np.average(med_out['beta_out'] / med_out['beta_med'], weights=weights_b)
    se_b = np.sqrt(1 / np.sum(weights_b))
    p_b = 2 * (1 - stats.norm.cdf(np.abs(b / se_b))) if se_b > 0 else 1.0

    # 间接效应
    indirect = a * b
    # Bootstrap CI
    np.random.seed(123)
    boot_indirect = []
    for _ in range(n_bootstrap):
        boot_exp_med = exp_med.sample(frac=1, replace=True)
        w_a_boot = 1 / (boot_exp_med['se_med'] ** 2 / boot_exp_med['beta_exp'] ** 2 +
                        boot_exp_med['beta_med'] ** 2 * boot_exp_med['se_exp'] ** 2 / boot_exp_med['beta_exp'] ** 4)
        a_boot = np.average(boot_exp_med['beta_med'] / boot_exp_med['beta_exp'], weights=w_a_boot)
        boot_med_out = med_out.sample(frac=1, replace=True)
        w_b_boot = 1 / (boot_med_out['se_out'] ** 2 / boot_med_out['beta_med'] ** 2 +
                        boot_med_out['beta_out'] ** 2 * boot_med_out['se_med'] ** 2 / boot_med_out['beta_med'] ** 4)
        b_boot = np.average(boot_med_out['beta_out'] / boot_med_out['beta_med'], weights=w_b_boot)
        boot_indirect.append(a_boot * b_boot)
    ci_lower, ci_upper = np.percentile(boot_indirect, [2.5, 97.5])

    # 总效应
    exp_out = pd.merge(exposure_df, outcome_df, on='SNP', suffixes=('_exp', '_out'), how='inner')
    exp_out = exp_out[(exp_out['beta_exp'] != 0) & (exp_out['beta_out'] != 0)]
    if len(exp_out) >= 2:
        weights_total = 1 / (exp_out['se_out'] ** 2 / exp_out['beta_exp'] ** 2 +
                             exp_out['beta_out'] ** 2 * exp_out['se_exp'] ** 2 / exp_out['beta_exp'] ** 4)
        total = np.average(exp_out['beta_out'] / exp_out['beta_exp'], weights=weights_total)
    else:
        total = np.nan

    direct = total - indirect if not np.isnan(total) else np.nan

    return {
        'a': a, 'se_a': se_a, 'p_a': p_a,
        'b': b, 'se_b': se_b, 'p_b': p_b,
        'indirect': indirect,
        'indirect_CI_lower': ci_lower,
        'indirect_CI_upper': ci_upper,
        'total_effect': total,
        'direct_effect': direct,
        'n_snps_XM': len(exp_med),
        'n_snps_MY': len(med_out)
    }


def main():
    # 读取结局数据
    if not OUTCOME_FILE.exists():
        print("❌ 结局文件不存在，请先运行 prepare_mr_data.py")
        return
    outcome = pd.read_csv(OUTCOME_FILE, sep='\t')
    outcome = outcome[['SNP', 'beta', 'se']].rename(
        columns={'beta': 'beta_out', 'se': 'se_out'})

    # 对 LUAD 和 LUSC 分别处理
    for cancer in ['LUAD', 'LUSC']:
        exp_file = TCGA_DIR / f"{cancer}_cis_target_eqtl.tsv"
        if not exp_file.exists():
            print(f"⚠️ {cancer} eQTL 文件不存在，跳过")
            continue
        exp = pd.read_csv(exp_file, sep='\t')
        exp['gene'] = exp['gene'].str.split('|').str[0]
        exp = exp[exp['gene'] == 'IL18'].rename(
            columns={'beta': 'beta_exp', 'se': 'se_exp'})
        print(f"\n📌 {cancer} IL18 eQTL SNPs: {len(exp)}")

        for med_name, med_file in MEDIATOR_FILES.items():
            if not med_file.exists():
                print(f"  ⚠️ {med_name} 中介文件不存在，跳过")
                continue
            print(f"\n  🔍 中介分析：{cancer} IL18 → {med_name}")
            med = pd.read_csv(med_file, sep='\t')
            # 确保列名正确
            if 'beta' not in med.columns or 'se' not in med.columns:
                print(f"    ❌ {med_name} 文件缺少 beta/se 列，请检查格式")
                continue
            med = med[['SNP', 'beta', 'se']].rename(
                columns={'beta': 'beta_med', 'se': 'se_med'})

            res = mediation_analysis_improved(exp, med, outcome)
            if res:
                print(f"    a (IL18->{med_name}): β={res['a']:.4f}, p={res['p_a']:.4e}")
                print(f"    b ({med_name}->Lung): β={res['b']:.4f}, p={res['p_b']:.4e}")
                print(
                    f"    Indirect effect = {res['indirect']:.6f} (95% CI: {res['indirect_CI_lower']:.6f}, {res['indirect_CI_upper']:.6f})")
                prop = res['indirect'] / res['total_effect'] * 100 if not np.isnan(res['total_effect']) else np.nan
                print(f"    Proportion mediated = {prop:.1f}%")

                # 保存结果
                res['cancer'] = cancer
                res['mediator'] = med_name
                df_res = pd.DataFrame([res])
                out_file = OUTPUT_DIR / f"{cancer}_IL18_mediation_{med_name}.csv"
                df_res.to_csv(out_file, index=False)
                print(f"    ✅ 结果已保存至 {out_file}")
            else:
                print("    ❌ 中介分析失败（SNP不足）")


if __name__ == "__main__":
    main()