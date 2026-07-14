import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BERT = pd.read_csv("data/out/bertimbau/bertimbau_global_metrics.csv", index_col=0)
QWEN = pd.read_csv("data/out/qwen_fewshot/qwen_global_metrics.csv", index_col=0)

metrics = ["f1_macro", "f1_micro", "f1_weighted", "roc_auc_macro"]
comp = pd.DataFrame({
    "BERTimbau_mean": BERT[metrics].mean(), "BERTimbau_std": BERT[metrics].std(),
    "QWEN_mean":      QWEN[metrics].mean(), "QWEN_std":      QWEN[metrics].std(),
}).round(4)
comp.to_csv("data/out/qwen_fewshot/comparison_summary.csv")
print(comp)

x = np.arange(len(metrics)); w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, comp["BERTimbau_mean"], w, yerr=comp["BERTimbau_std"],
       label="BERTimbau (fine-tuned)", capsize=4, color="#4C72B0", alpha=.85)
ax.bar(x + w/2, comp["QWEN_mean"], w, yerr=comp["QWEN_std"],
       label="QWEN (few-shot)", capsize=4, color="#DD8452", alpha=.85)
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0, 1)
ax.set_title("BERTimbau × QWEN few-shot — GoEmotions"); ax.legend()
plt.tight_layout(); plt.savefig("data/out/qwen_fewshot/comparison.png", dpi=150)
print("Figura de comparação salva.")