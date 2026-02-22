#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略B数据准备脚本 - Python版本
将各种格式的遗传数据转换为标准MR格式
"""

import pandas as pd
import numpy as np
import gzip
import tarfile
import os
import re
from pathlib import Path
import sys

print("=" * 70)
print("策略B数据准备：Python数据处理脚本")
print("=" * 70)


class MRDataPreparer:
    """MR数据准备器"""

    def __init__(self, output_dir="MR_Data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def prepare_il18_exposure(self, il18_file="IL18.data.gz"):
        """准备IL18暴露数据"""
        print(f"\n1. 准备IL18暴露数据 ({il18_file})...")

        if not Path(il18_file).exists():
            print(f"  ❌ 文件不存在: {il18_file}")
            return None

        try:
            # 读取IL18数据（空格分隔）
            df = pd.read_csv(il18_file, sep=r'\s+', engine='python')
            print(f"  原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

            # 查看列名
            print(f"  列名: {list(df.columns)}")

            # 标准化列名
            column_mapping = {
                'MarkerName': 'SNP',
                'Chromosome': 'chr',
                'Position': 'pos',
                'EffectAllele': 'effect_allele',
                'OtherAllele': 'other_allele',
                'Effect': 'beta',
                'StdErr': 'se',
                'P.value': 'pval'
            }

            # 重命名列
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]

            # 选择标准列
            required_cols = ['SNP', 'effect_allele', 'other_allele', 'beta', 'se', 'pval']
            optional_cols = ['chr', 'pos']

            available_cols = []
            for col in required_cols + optional_cols:
                if col in df.columns:
                    available_cols.append(col)

            df_final = df[available_cols].copy()

            # 筛选显著SNP (P < 5e-6)
            if 'pval' in df_final.columns:
                df_final = df_final[df_final['pval'] < 5e-6].copy()
                print(f"  筛选后(P < 5e-6): {len(df_final)} 个SNP")

            # 计算F统计量
            if 'beta' in df_final.columns and 'se' in df_final.columns:
                df_final['f_stat'] = (df_final['beta'] / df_final['se']) ** 2
                avg_f = df_final['f_stat'].mean()
                print(f"  平均F统计量: {avg_f:.2f}")

            # 保存
            output_file = self.output_dir / "IL18_exposure.tsv"
            df_final.to_csv(output_file, sep='\t', index=False)
            print(f"  ✅ 已保存: {output_file}")

            return output_file

        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_vcf_to_mr(self, vcf_file, output_name, expected_format=None):
        """转换VCF文件为MR格式"""
        print(f"\n转换VCF文件: {vcf_file}")

        if not Path(vcf_file).exists():
            print(f"  ❌ 文件不存在: {vcf_file}")
            return None

        try:
            data = []
            snp_count = 0

            with gzip.open(vcf_file, 'rt') as f:
                for line in f:
                    if line.startswith('#'):
                        continue

                    fields = line.strip().split('\t')

                    # 基本列
                    chrom = fields[0]
                    pos = fields[1]
                    rsid = fields[2] if fields[2] != '.' else f"{chrom}:{pos}"
                    ref = fields[3]
                    alt = fields[4]

                    # 解析FORMAT和样本数据
                    if len(fields) >= 10:
                        format_col = fields[8]
                        sample_col = fields[9]

                        # 解析样本数据
                        format_keys = format_col.split(':')
                        sample_values = sample_col.split(':')

                        if len(format_keys) == len(sample_values):
                            sample_dict = dict(zip(format_keys, sample_values))

                            # 提取效应值、标准误、P值
                            es = sample_dict.get('ES')
                            se = sample_dict.get('SE')
                            lp = sample_dict.get('LP')  # -log10(p)
                            af = sample_dict.get('AF')

                            if es and se and lp:
                                try:
                                    beta = float(es)
                                    se_val = float(se)
                                    lp_val = float(lp)
                                    pval = 10 ** (-lp_val)

                                    # 效应等位基因频率
                                    eaf = float(af) if af and af != '.' else None

                                    # 确定效应等位基因
                                    effect_allele = alt
                                    other_allele = ref

                                    data.append({
                                        'SNP': rsid,
                                        'chr': chrom,
                                        'pos': pos,
                                        'effect_allele': effect_allele,
                                        'other_allele': other_allele,
                                        'beta': beta,
                                        'se': se_val,
                                        'pval': pval,
                                        'eaf': eaf
                                    })

                                    snp_count += 1

                                except (ValueError, TypeError) as e:
                                    continue

                    # 进度显示
                    if snp_count % 100000 == 0 and snp_count > 0:
                        print(f"  已处理 {snp_count:,} 个SNP...")

            if data:
                df = pd.DataFrame(data)

                # 计算F统计量
                df['f_stat'] = (df['beta'] / df['se']) ** 2

                # 保存
                output_file = self.output_dir / f"{output_name}.tsv"
                df.to_csv(output_file, sep='\t', index=False)

                avg_f = df['f_stat'].mean()
                print(f"  ✅ 转换完成: {output_file}")
                print(f"    SNP数量: {len(df):,}")
                print(f"    平均F统计量: {avg_f:.2f}")
                print(f"    P值范围: {df['pval'].min():.2e} - {df['pval'].max():.2e}")

                return output_file
            else:
                print(f"  ❌ 没有提取到有效数据")
                return None

        except Exception as e:
            print(f"  ❌ 转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def prepare_all_data(self):
        """准备所有数据"""
        print("\n" + "=" * 70)
        print("开始准备所有MR数据")
        print("=" * 70)

        results = {}

        # 1. IL18暴露数据
        il18_file = self.prepare_il18_exposure()
        if il18_file:
            results['IL18_exposure'] = str(il18_file)

        # 2. 肺癌结局数据 (UK Biobank)
        if Path("ukb-b-14521.vcf.gz").exists():
            lung_file = self.convert_vcf_to_mr(
                "ukb-b-14521.vcf.gz",
                "lung_cancer_outcome"
            )
            if lung_file:
                results['lung_cancer'] = str(lung_file)

        # 3. 淋巴细胞计数 (中介)
        if Path("ieu-a-985.vcf.gz").exists():
            lymph_file = self.convert_vcf_to_mr(
                "ieu-a-985.vcf.gz",
                "lymphocyte_mediator"
            )
            if lymph_file:
                results['lymphocyte'] = str(lymph_file)

        # 4. 单核细胞计数 (中介)
        if Path("ieu-a-987.vcf.gz").exists():
            mono_file = self.convert_vcf_to_mr(
                "ieu-a-987.vcf.gz",
                "monocyte_mediator"
            )
            if mono_file:
                results['monocyte'] = str(mono_file)

        # 5. 其他数据
        vcf_files = {
            "ebi-a-GCST90018655.vcf.gz": "inflammation_mediator",
            "finn-b-CD2_BENIGN_BRONCHUS_LUNG_EXALLC.vcf.gz": "benign_lung_control"
        }

        for vcf_file, output_name in vcf_files.items():
            if Path(vcf_file).exists():
                result_file = self.convert_vcf_to_mr(vcf_file, output_name)
                if result_file:
                    results[output_name] = str(result_file)

        # 保存数据清单
        self.save_data_inventory(results)

        return results

    def save_data_inventory(self, results):
        """保存数据清单"""
        inventory_file = self.output_dir / "data_inventory.txt"

        with open(inventory_file, 'w') as f:
            f.write("策略B - MR数据清单\n")
            f.write("=" * 50 + "\n\n")

            f.write("文件列表:\n")
            for data_type, file_path in results.items():
                f.write(f"{data_type:30} {file_path}\n")

            f.write("\n\nR分析代码示例:\n")
            f.write("=" * 50 + "\n")
            f.write(self.generate_r_code())

        print(f"\n✅ 数据清单已保存: {inventory_file}")

    def generate_r_code(self):
        """生成R分析代码模板"""
        r_code = '''
# ============================================================================
# 策略B MR分析 - R脚本
# 使用Python预处理后的数据
# ============================================================================

# 安装必要包（如果未安装）
# install.packages(c("TwoSampleMR", "MRPRESSO", "coloc", "ggplot2", "dplyr"))

# 加载包
library(TwoSampleMR)
library(MRPRESSO)
library(coloc)
library(ggplot2)
library(dplyr)

# 设置工作目录
setwd("C:/Users/wswan/PycharmProjects/PythonProject2")

# 读取Python预处理的数据
read_mr_data <- function(filename, type = "exposure", ...) {
  data <- read_exposure_data(
    filename = filename,
    sep = "\\t",
    snp_col = "SNP",
    beta_col = "beta",
    se_col = "se",
    effect_allele_col = "effect_allele",
    other_allele_col = "other_allele",
    pval_col = "pval",
    ...
  )
  return(data)
}

# 1. 读取IL18暴露数据
exposure_il18 <- read_mr_data("MR_Data/IL18_exposure.tsv")
exposure_il18$exposure <- "IL-18"
exposure_il18$id.exposure <- "IL18"

# 2. 读取肺癌结局数据
outcome_lung <- read_mr_data("MR_Data/lung_cancer_outcome.tsv", type = "outcome")
outcome_lung$outcome <- "Lung Cancer"
outcome_lung$id.outcome <- "LUNG_CANCER"

# 3. 数据协调
dat <- harmonise_data(exposure_il18, outcome_lung, action = 2)

# 4. MR分析
results <- mr(dat, method_list = c("mr_ivw", "mr_egger", "mr_weighted_median"))

# 5. 显示结果
print(results)

# 6. 敏感性分析
pleiotropy <- mr_pleiotropy_test(dat)
heterogeneity <- mr_heterogeneity(dat)
print(pleiotropy)
print(heterogeneity)

# 7. 可视化
scatter_plot <- mr_scatter_plot(results, dat)
ggsave("mr_scatter_plot.png", scatter_plot[[1]], width = 8, height = 6, dpi = 300)

# 8. 中介分析（如果需要）
# 读取淋巴细胞数据
if(file.exists("MR_Data/lymphocyte_mediator.tsv")) {
  outcome_lymph <- read_mr_data("MR_Data/lymphocyte_mediator.tsv", type = "outcome")
  outcome_lymph$outcome <- "Lymphocyte Count"
  outcome_lymph$id.outcome <- "LYMPH"

  dat_mediator <- harmonise_data(exposure_il18, outcome_lymph, action = 2)
  mr_mediator <- mr(dat_mediator, method_list = "mr_ivw")
  print(mr_mediator)
}
'''
        return r_code


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("MR数据准备流程")
    print("=" * 70)

    # 创建数据准备器
    preparer = MRDataPreparer()

    # 准备所有数据
    results = preparer.prepare_all_data()

    print("\n" + "=" * 70)
    print("数据准备完成！")
    print("=" * 70)

    if results:
        print("\n已准备的数据文件:")
        for data_type, file_path in results.items():
            print(f"  {data_type:30} -> {file_path}")

        print("\n下一步:")
        print("1. 运行R脚本进行MR分析")
        print("2. 检查MR_Data/data_inventory.txt获取R代码示例")
        print("3. 调整R脚本中的文件路径和分析参数")
    else:
        print("❌ 没有成功准备任何数据")

    return results


if __name__ == "__main__":
    main()