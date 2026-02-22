import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['IL1B (MVMR)', 'IL18 (MVMR)', 'TNFa (MVMR)', 'Inflammation (IVW)', 'Lymphocyte (IVW)', 'Monocyte (IVW)']
beta = [-0.00142, 0.00001, -0.000001, 0.000026, 0.00232, 0.00349]
ci_lower = [-0.00573, -0.00026, -0.00127, 0.000018, 0.00228, 0.00345]
ci_upper = [0.00289, 0.00028, 0.00126, 0.000034, 0.00235, 0.00353]
colors = ['gray', 'gray', 'gray', 'steelblue', 'steelblue', 'steelblue']

y_pos = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(beta, y_pos, xerr=[np.array(beta)-np.array(ci_lower), np.array(ci_upper)-np.array(beta)],
            fmt='o', color='steelblue', ecolor='gray', capsize=3)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel('Effect size (β) for lung cancer')
ax.set_title('Figure 7. MR estimates for immune toxicity subtypes')
plt.tight_layout()
plt.savefig('subtype_mr_forest.png', dpi=300)