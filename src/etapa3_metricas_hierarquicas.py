"""
ETAPA 3 — Métricas hierárquicas / sensíveis à distância

Compara, sobre as predições OOF do BERTimbau @limiar:
  A) Macro-F1 tradicional (fino, 27)
  B) F1 após agrupamento (Ekman 6 e sentimento 3)
  C) métricas sensíveis à distância na árvore raiz->sentimento->Ekman->fina:
     - distância média de confusão (ultramétrica) vs baseline aleatório
     - taxa de erro perto/médio/longe
     - matriz de confusão ponderada pela distância
     - F1 hierárquico (hP/hR/hF) por aumento de ancestrais (Kiritchenko; CoPHE)

Ancoragem:
  Wu, Tygert & LeCun (2017)  -> árvore ULTRAMÉTRICA (folhas equidistantes da raiz);
                                base da distância d in {0,1/3,2/3,1}.
  Cao, Feng & An (2024)      -> tree distance loss: erro custa a distância na árvore;
                                aqui usamos a distância post-hoc (sem retreinar).
  Falis et al. (2021, CoPHE) -> avaliação hierárquica por profundidade/ancestrais;
                                base do hP/hR/hF e da variante ponderada por profundidade.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

ROOT = Path("")
DATA_DIR = ROOT / "data/treated"
OUT_BASE = ROOT / "data/out"
CACHE_DIR = OUT_BASE / "bertimbau/cache"
EXPORT_DIR = OUT_BASE / "etapa3_metricas_hierarquicas"
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

EKMAN_MAP = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude",
            "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}
SENTIMENT_MAP = {
    "positive": EKMAN_MAP["joy"],
    "negative": EKMAN_MAP["anger"] + EKMAN_MAP["disgust"] + EKMAN_MAP["fear"] + EKMAN_MAP["sadness"],
    "ambiguous": EKMAN_MAP["surprise"],
    "neutral": ["neutral"],
}
EMO_TO_EKMAN = {e: g for g, emos in EKMAN_MAP.items() for e in emos}
EMO_TO_SENT = {e: s for s, emos in SENTIMENT_MAP.items() for e in emos}
DIST_COLORS = {"perto": "#27ae60", "médio": "#e67e22", "longe": "#c0392b"}

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

# --- distância ultramétrica na árvore (folhas equidistantes da raiz) ---
def ultrametric_distance():
    D = np.ones((N_LABELS, N_LABELS))
    for i, a in enumerate(EMOTION_COLS):
        for j, b in enumerate(EMOTION_COLS):
            if a == b:
                D[i, j] = 0.0
            elif EMO_TO_EKMAN[a] == EMO_TO_EKMAN[b]:
                D[i, j] = 1/3
            elif EMO_TO_SENT[a] == EMO_TO_SENT[b]:
                D[i, j] = 2/3
            else:
                D[i, j] = 1.0
    return D

def dist_bucket(d):
    return "perto" if d <= 1/3 else ("médio" if d <= 2/3 else "longe")

# --- A e B: macro-F1 fino / Ekman / sentimento ---
def macro_f1_levels(Y_true, Y_pred):
    mapped = [e for e in EMOTION_COLS if e != "neutral"]
    idx = [EMOTION_COLS.index(e) for e in mapped]
    f1_fine = f1_score(Y_true[:, idx], Y_pred[:, idx], average="macro", zero_division=0)
    ek = [g for g in EKMAN_MAP if g != "neutral"]
    YTe = np.column_stack([merged_labels(Y_true, EKMAN_MAP[g]) for g in ek])
    YPe = np.column_stack([merged_labels(Y_pred, EKMAN_MAP[g]) for g in ek])
    f1_ek = f1_score(YTe, YPe, average="macro", zero_division=0)
    se = [s for s in SENTIMENT_MAP if s != "neutral"]
    YTs = np.column_stack([merged_labels(Y_true, SENTIMENT_MAP[s]) for s in se])
    YPs = np.column_stack([merged_labels(Y_pred, SENTIMENT_MAP[s]) for s in se])
    f1_sent = f1_score(YTs, YPs, average="macro", zero_division=0)
    return f1_fine, f1_ek, f1_sent

# --- confusão de substituição (mesma def da Etapa 2) ---
def substitution_confusion(Y_true, Y_pred):
    M = np.zeros((N_LABELS, N_LABELS), dtype=float)
    for i in range(N_LABELS):
        miss_i = (Y_true[:, i] == 1) & (Y_pred[:, i] == 0)
        for j in range(N_LABELS):
            if i == j:
                continue
            M[i, j] = (miss_i & (Y_pred[:, j] == 1)).sum()
    return M

# --- C1: distância média de confusão + taxa perto/médio/longe + baseline ---
def confusion_distance_stats(M, D, support):
    mass = M.sum()
    mean_dist = float((M * D).sum() / mass) if mass > 0 else np.nan
    buckets = {"perto": 0.0, "médio": 0.0, "longe": 0.0}
    for i in range(N_LABELS):
        for j in range(N_LABELS):
            if M[i, j] > 0 and i != j:
                buckets[dist_bucket(D[i, j])] += M[i, j]
    rates = {k: (v / mass if mass > 0 else np.nan) for k, v in buckets.items()}
    # baseline: "previu no lugar" ~ proporcional à frequência dos rótulos
    p = support / support.sum()
    exp_rand = float(sum(support[i] * sum(p[j] * D[i, j] for j in range(N_LABELS))
                         for i in range(N_LABELS)) / support.sum())
    return mean_dist, rates, exp_rand

# --- C2: matriz de confusão ponderada por distância, agregada por grupo ---
def grouped_confusion(M, level="ekman", weight_by_distance=False, D=None):
    keys = ([g for g in EKMAN_MAP] if level == "ekman" else [s for s in SENTIMENT_MAP])
    to = EMO_TO_EKMAN if level == "ekman" else EMO_TO_SENT
    idx = {k: n for n, k in enumerate(keys)}
    G = np.zeros((len(keys), len(keys)))
    for i, a in enumerate(EMOTION_COLS):
        for j, b in enumerate(EMOTION_COLS):
            if i == j:
                continue
            w = M[i, j] * (D[i, j] if weight_by_distance else 1.0)
            G[idx[to[a]], idx[to[b]]] += w
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-9)
    return pd.DataFrame(Gn, index=keys, columns=keys)

# --- C3: F1 hierárquico por aumento de ancestrais (Kiritchenko / CoPHE) ---
def build_augmentation():
    nodes = list(EMOTION_COLS)
    nodes += [f"EK::{g}" for g in EKMAN_MAP]
    nodes += [f"SE::{s}" for s in SENTIMENT_MAP]
    nidx = {n: i for i, n in enumerate(nodes)}
    depth = np.zeros(len(nodes))  # sentimento=1, ekman=2, fina=3 (peso p/ CoPHE)
    for n in nodes:
        depth[nidx[n]] = 3 if n in EMOTION_COLS else (2 if n.startswith("EK::") else 1)
    Aug = np.zeros((N_LABELS, len(nodes)))
    for i, e in enumerate(EMOTION_COLS):
        Aug[i, nidx[e]] = 1
        Aug[i, nidx[f"EK::{EMO_TO_EKMAN[e]}"]] = 1
        Aug[i, nidx[f"SE::{EMO_TO_SENT[e]}"]] = 1
    return Aug, depth

def hierarchical_prf(Y_true, Y_pred, depth_weighted=False):
    Aug, depth = build_augmentation()
    Tn = (Y_true @ Aug) > 0
    Pn = (Y_pred @ Aug) > 0
    w = depth if depth_weighted else np.ones_like(depth)
    inter = ((Tn & Pn) * w).sum()
    sp = (Pn * w).sum(); st = (Tn * w).sum()
    hP = inter / sp if sp > 0 else 0.0
    hR = inter / st if st > 0 else 0.0
    hF = 2*hP*hR/(hP+hR) if (hP+hR) > 0 else 0.0
    return float(hP), float(hR), float(hF)

# --- gráficos ---
def plot_distance_heatmap(D, path):
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(D, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(N_LABELS)); ax.set_xticklabels(EMOTION_COLS, rotation=90, fontsize=7)
    ax.set_yticks(range(N_LABELS)); ax.set_yticklabels(EMOTION_COLS, fontsize=7)
    ax.set_title("Distância ultramétrica entre emoções (0=igual, 1/3=perto, 2/3=médio, 1=longe)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); plt.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)

def plot_near_far_stacked(M, D, support, path):
    rows = []
    for i, e in enumerate(EMOTION_COLS):
        b = {"perto": 0.0, "médio": 0.0, "longe": 0.0}
        for j in range(N_LABELS):
            if M[i, j] > 0 and i != j:
                b[dist_bucket(D[i, j])] += M[i, j]
        tot = sum(b.values())
        if tot > 0:
            rows.append((e, b["perto"]/tot, b["médio"]/tot, b["longe"]/tot, tot))
    d = pd.DataFrame(rows, columns=["emocao", "perto", "médio", "longe", "tot"])
    d = d.sort_values("perto")
    fig, ax = plt.subplots(figsize=(9, 10))
    left = np.zeros(len(d))
    for col in ["perto", "médio", "longe"]:
        ax.barh(d["emocao"], d[col], left=left, color=DIST_COLORS[col], label=col, alpha=0.9)
        left += d[col].values
    ax.set_xlim(0, 1); ax.set_xlabel("fração da massa de erro por distância")
    ax.set_title("Onde caem os erros de cada emoção (perto/médio/longe na taxonomia)")
    ax.legend(loc="lower right"); plt.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    return d

def plot_error_profile(M, D, support, emo, path):
    i = EMOTION_COLS.index(emo)
    order = np.argsort(-M[i])
    tgts = [EMOTION_COLS[j] for j in order if M[i, j] > 0][:8]
    vals = [M[i, j] for j in order if M[i, j] > 0][:8]
    cols = [DIST_COLORS[dist_bucket(D[i, EMOTION_COLS.index(t)])] for t in tgts]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(tgts)), vals, color=cols, alpha=0.9)
    ax.set_xticks(range(len(tgts))); ax.set_xticklabels(tgts, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("nº de substituições no erro")
    ax.set_title(f"Erros de '{emo}' (verde=perto, laranja=médio, vermelho=longe)")
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def plot_grouped_confusion(G, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(G.values, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(G))); ax.set_xticklabels(G.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(G))); ax.set_yticklabels(G.index, fontsize=8)
    for r in range(len(G)):
        for c in range(len(G)):
            ax.text(c, r, f"{G.values[r,c]:.2f}", ha="center", va="center",
                    fontsize=7, color="#333" if G.values[r, c] < 0.5 else "#fff")
    ax.set_xlabel("previsto no lugar"); ax.set_ylabel("real")
    ax.set_title(title); plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def main():
    print("Carregando folds + probabilidades OOF em cache…")
    Y, P = load_folds()
    Y_true, Y_pred = build_oof_predictions(Y, P)
    support = Y_true.sum(0).astype(float)
    print(f"  N amostras OOF = {Y_true.shape[0]} | labels = {N_LABELS}")

    D = ultrametric_distance()
    M = substitution_confusion(Y_true, Y_pred)

    f1_fine, f1_ek, f1_sent = macro_f1_levels(Y_true, Y_pred)
    mean_dist, rates, exp_rand = confusion_distance_stats(M, D, support)
    hP, hR, hF = hierarchical_prf(Y_true, Y_pred)
    hPw, hRw, hFw = hierarchical_prf(Y_true, Y_pred, depth_weighted=True)

    G_ek = grouped_confusion(M, "ekman")
    G_se = grouped_confusion(M, "sentimento")

    resumo = pd.DataFrame([
        {"metrica": "A) Macro-F1 fino (27)", "valor": round(f1_fine, 4)},
        {"metrica": "B) Macro-F1 Ekman (6)", "valor": round(f1_ek, 4)},
        {"metrica": "B) Macro-F1 sentimento (3)", "valor": round(f1_sent, 4)},
        {"metrica": "C) dist. média de confusão", "valor": round(mean_dist, 4)},
        {"metrica": "C) dist. esperada (aleatório)", "valor": round(exp_rand, 4)},
        {"metrica": "C) taxa erro PERTO", "valor": round(rates["perto"], 4)},
        {"metrica": "C) taxa erro MÉDIO", "valor": round(rates["médio"], 4)},
        {"metrica": "C) taxa erro LONGE", "valor": round(rates["longe"], 4)},
        {"metrica": "C) F1 hierárquico hF", "valor": round(hF, 4)},
        {"metrica": "C) hF ponderado por profundidade", "valor": round(hFw, 4)},
    ])

    resumo.to_csv(EXPORT_DIR / "etapa3_resumo_metricas.csv", index=False)
    G_ek.round(4).to_csv(EXPORT_DIR / "etapa3_confusao_ekman.csv")
    G_se.round(4).to_csv(EXPORT_DIR / "etapa3_confusao_sentimento.csv")
    pd.DataFrame(D, index=EMOTION_COLS, columns=EMOTION_COLS)\
        .to_csv(EXPORT_DIR / "etapa3_distancia_ultrametrica.csv")
    pd.DataFrame(M * D, index=EMOTION_COLS, columns=EMOTION_COLS)\
        .to_csv(EXPORT_DIR / "etapa3_confusao_ponderada.csv")
    with pd.ExcelWriter(EXPORT_DIR / "etapa3_metricas.xlsx") as xw:
        resumo.to_excel(xw, sheet_name="resumo", index=False)
        G_ek.round(4).to_excel(xw, sheet_name="confusao_ekman")
        G_se.round(4).to_excel(xw, sheet_name="confusao_sentimento")

    plot_distance_heatmap(D, EXPORT_DIR / "etapa3_distancia_heatmap.png")
    perfil = plot_near_far_stacked(M, D, support, EXPORT_DIR / "etapa3_near_far.png")
    plot_grouped_confusion(G_ek, "Confusão entre grupos de Ekman (linha=real)",
                           EXPORT_DIR / "etapa3_confusao_ekman.png")
    plot_grouped_confusion(G_se, "Confusão entre sentimentos (linha=real)",
                           EXPORT_DIR / "etapa3_confusao_sentimento.png")
    # exemplos: as 2 emoções que mais erram perto e as 2 que mais erram longe
    if len(perfil):
        for e in list(perfil.sort_values("perto", ascending=False)["emocao"].head(2)) + \
                 list(perfil.sort_values("longe", ascending=False)["emocao"].head(2)):
            plot_error_profile(M, D, support, e, EXPORT_DIR / f"etapa3_perfil_{e}.png")

    print("\n" + "="*70)
    print("A) Macro-F1 fino (27)        = %.4f" % f1_fine)
    print("B) Macro-F1 Ekman (6)        = %.4f   (Δ vs fino = %+.4f)" % (f1_ek, f1_ek - f1_fine))
    print("B) Macro-F1 sentimento (3)   = %.4f   (Δ vs fino = %+.4f)" % (f1_sent, f1_sent - f1_fine))
    print("-"*70)
    print("C) distância média de confusão = %.4f  (aleatório = %.4f)" % (mean_dist, exp_rand))
    print("   erros PERTO=%.1f%%  MÉDIO=%.1f%%  LONGE=%.1f%%"
          % (rates["perto"]*100, rates["médio"]*100, rates["longe"]*100))
    print("C) F1 hierárquico hF = %.4f | hP=%.4f hR=%.4f | hF(profund.)=%.4f"
          % (hF, hP, hR, hFw))
    print("="*70)
    print("\nConfusão entre grupos de Ekman (linha=real, coluna=previsto):")
    print(G_ek.round(3).to_string())
    print(f"\n✓ Exportado em {EXPORT_DIR.resolve()}")

if __name__ == "__main__":
    main()
