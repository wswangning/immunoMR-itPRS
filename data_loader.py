print("\n" + "=" * 80)
print("策略B - 完整数据清单")
print("=" * 80)

data_inventory = {
    "暴露数据 (Exposure)": [
        ("IL18 pQTL", "IL18.data.gz", "炎症因子蛋白水平"),
        ("IL1B pQTL", "IL1b.data.gz", "炎症因子蛋白水平"),
        ("TNFa pQTL", "TNFa.data.gz", "炎症因子蛋白水平"),
        ("NLRP3通路eQTL", "NLRP3_pathway_eQTLs.tsv", "基因表达水平"),
        ("cis-eQTL全集", "cis-eQTL-SMR/", "多组织基因表达")
    ],

    "中介数据 (Mediator)": [
        ("淋巴细胞计数", "lymphocyte_count_GWAS_for_MR.tsv", "免疫细胞特征"),
        ("淋巴细胞计数(原始)", "lymph.tsv.gz", "免疫细胞特征"),
        ("白细胞计数", "wbc.tsv.gz", "免疫细胞特征"),
        ("中性粒细胞", "neut.tsv.gz", "免疫细胞特征")
    ],

    "结局数据 (Outcome)": [
        ("肺癌GWAS", "lung_cancer_GWAS_for_MR.tsv", "主要结局"),
        ("肺癌亚型?", "待确认", "肺腺癌/鳞癌")
    ]
}

for category, items in data_inventory.items():
    print(f"\n{category}:")
    print("-" * 60)
    for name, file, description in items:
        status = "✅ 已准备" if os.path.exists(file) else "⚠️  需处理"
        print(f"  {name:20} {file:35} {description:25} {status}")