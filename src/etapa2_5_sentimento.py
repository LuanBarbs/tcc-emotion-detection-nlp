"""
ETAPA 2.5 - Métricas de sentimento (positive / negative / ambiguous)

Terceiro nível da hierarquia (fino -> Ekman -> sentimento).
Reaproveita as probabilidades OOF do baseline e os limiares calibrados @limiar.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, f1_score

warnings.filterwarnings("ignore")

ROOT = Path("")
DATA_DIR = ROOT / "data/treated"
OUT_BASE = ROOT / "data/out"
CACHE_DIR = OUT_BASE / "bertimbau/cache"
EXPORT_DIR = OUT_BASE / "etapa2_5_sentimento"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

N_OUTER_FOLDS = 5
FIXED_THRESHOLD = 0.5
THRESH_GRID = np.linspace(0.05, 0.95, 19)

EMOTION_COLS = [
    'admiration', 'amusement', 'anger', 'annoyance',
    'approval', 'caring', 'confusion', 'curiosity', 'desire',
    'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
    'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]
N_LABELS = len(EMOTION_COLS)

SENTIMENT_MAP = {
    "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism",
                 "caring", "pride", "admiration", "gratitude", "relief", "approval"],
    "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment",
                 "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
}
EMO_TO_SENT = {e: s for s, emos in SENTIMENT_MAP.items() for e in emos}
SENT_COLORS = {"positive": "#2ecc71", "negative": "#e74c3c", "ambiguous": "#f1c40f"}

def prepare_Y(df):
    present = [c for c in EMOTION_COLS if c in df.columns]
    assert len(present) == N_LABELS, f"Faltam labels: {set(EMOTION_COLS) - set(present)}"
    return df[present].values.astype(int)

def load_folds():
    Y, P = {}, {}
    for k in range(1, N_OUTER_FOLDS + 1):
        df = pd.read_parquet(DATA_DIR / f"fold_{k}" / "data.parquet")
        Y[k] = prepare_Y(df)
        p_path = CACHE_DIR / f"P_bertimbau_fold{k}.npy"
        if not p_path.exists():
            raise FileNotFoundError(f"Cache OOF não encontrado: {p_path}")
        P[k] = np.load(p_path)
    return Y, P

def calibrate_thresholds(P_cal, Y_cal, grid=THRESH_GRID):
    thr = np.full(N_LABELS, FIXED_THRESHOLD)
    for j in range(N_LABELS):
        y = Y_cal[:, j]
        if y.sum() == 0:
            continue
        pred = P_cal[:, j][:, None] >= grid[None, :]
        yb = (y == 1)[:, None]
        tp = (pred & yb).sum(0); fp = (pred & ~yb).sum(0); fn = (~pred & yb).sum(0)
        f1 = 2*tp / np.maximum(2*tp + fp + fn, 1e-12)
        thr[j] = float(grid[int(np.argmax(f1))])
    return thr

def build_oof_predictions(Y, P):
    Yt, Yp = [], []
    for k in range(1, N_OUTER_FOLDS + 1):
        others = [j for j in range(1, N_OUTER_FOLDS + 1) if j != k]
        thr = calibrate_thresholds(np.vstack([P[j] for j in others]),
                                   np.vstack([Y[j] for j in others]))
        Yt.append(Y[k]); Yp.append((P[k] >= thr[None, :]).astype(int))
    return np.vstack(Yt), np.vstack(Yp)

def merged_labels(Y, emos):
    idx = [EMOTION_COLS.index(e) for e in emos]
    return Y[:, idx].max(axis=1)

def sentiment_matrices(Y):
    sents = list(SENTIMENT_MAP.keys())
    return np.column_stack([merged_labels(Y, SENTIMENT_MAP[s]) for s in sents]), sents

def per_sentiment_table(Y_true, Y_pred):
    YT, sents = sentiment_matrices(Y_true)
    YP, _ = sentiment_matrices(Y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        YT, YP, average=None, zero_division=0, labels=range(len(sents)))
    support = YT.sum(0)
    N = YT.shape[0]
    rows = []
    for j, s in enumerate(sents):
        rows.append({
            "sentimento": s,
            "n_emocoes": len(SENTIMENT_MAP[s]),
            "suporte": int(support[j]),
            "frequencia": support[j] / N,
            "precision": prec[j],
            "recall": rec[j],
            "f1": f1[j],
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

def macro_f1_fine_vs_sentiment(Y_true, Y_pred):
    mapped = [e for e in EMOTION_COLS if e in EMO_TO_SENT]
    idx = [EMOTION_COLS.index(e) for e in mapped]
    f1_fine = f1_score(Y_true[:, idx], Y_pred[:, idx], average="macro", zero_division=0)
    YT, _ = sentiment_matrices(Y_true)
    YP, _ = sentiment_matrices(Y_pred)
    f1_sent = f1_score(YT, YP, average="macro", zero_division=0)
    f1_sent_micro = f1_score(YT, YP, average="micro", zero_division=0)
    return f1_fine, f1_sent, f1_sent_micro, f1_sent - f1_fine

def plot_sentiment_f1(sent_df, path):
    d = sent_df.sort_values("f1")
    colors = [SENT_COLORS[s] for s in d["sentimento"]]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(d["sentimento"], d["f1"], color=colors, alpha=0.9)
    ax.set_xlabel("F1 (OOF, BERTimbau @limiar)"); ax.set_xlim(0, 1)
    ax.set_title("F1 por sentimento")
    for i, v in enumerate(d["f1"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", ls=":", alpha=0.4); plt.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)

def main():
    print("Carregando folds + probabilidades OOF em cache…")
    Y, P = load_folds()
    Y_true, Y_pred = build_oof_predictions(Y, P)
    print(f"  N amostras OOF = {Y_true.shape[0]} | labels = {N_LABELS}")

    sent_df = per_sentiment_table(Y_true, Y_pred)
    f1f, f1s, f1s_micro, d = macro_f1_fine_vs_sentiment(Y_true, Y_pred)

    map_df = pd.DataFrame([{"emocao": e, "sentimento": s}
                           for s, emos in SENTIMENT_MAP.items() for e in emos])

    sent_df.to_csv(EXPORT_DIR / "etapa2_5_por_sentimento.csv", index=False)
    map_df.to_csv(EXPORT_DIR / "etapa2_5_mapa_emocao_sentimento.csv", index=False)
    with pd.ExcelWriter(EXPORT_DIR / "etapa2_5_sentimento.xlsx") as xw:
        sent_df.round(4).to_excel(xw, sheet_name="por_sentimento", index=False)
        map_df.to_excel(xw, sheet_name="mapa_emocao_sentimento", index=False)
    plot_sentiment_f1(sent_df, EXPORT_DIR / "etapa2_5_f1_sentimento.png")

    print("\n" + "="*70)
    print("MACRO-F1  fino (27, sem neutral) = %.4f  |  sentimento (3) = %.4f  |  Δ = %+.4f"
          % (f1f, f1s, d))
    print("F1-micro sentimento = %.4f" % f1s_micro)
    print("="*70)
    print("\nF1 por sentimento:")
    print(sent_df.round(4).to_string(index=False))
    print(f"\n✓ Exportado em {EXPORT_DIR.resolve()}")

if __name__ == "__main__":
    main()