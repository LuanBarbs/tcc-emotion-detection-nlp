import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def save_plots(df_results, f1_per_label_folds, f1_per_label_mean, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metric_plot = ["f1_macro", "f1_micro", "f1_weighted", "roc_auc_macro"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    ax = axes[0]
    x = np.arange(len(df_results))
    width = 0.20
    for i, (met, col) in enumerate(zip(metric_plot, colors)):
        ax.bar(x + i * width, df_results[met], width, label=met, color=col, alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"Fold {k}" for k in df_results.index])
    ax.set_ylim(0, 1)
    ax.set_title("QWEN few-shot — Métricas por Outer Fold")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Score")

    ax2 = axes[1]
    f1_per_label_mean.plot.barh(ax=ax2, color="#4C72B0", alpha=0.85)
    ax2.set_title("F1 por Label (média entre folds)")
    ax2.set_xlabel("F1 Score")
    ax2.set_xlim(0, 1)
    ax2.axvline(f1_per_label_mean.mean(), color="red", linestyle="--",
                label=f"Média={f1_per_label_mean.mean():.3f}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = out_dir / "qwen_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figura salva em {out}")