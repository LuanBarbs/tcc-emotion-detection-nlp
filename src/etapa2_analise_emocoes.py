"""
ETAPA 2 — Análise completa das 28 emoções (BERTimbau @limiar calibrado)

Objetivo: para CADA emoção, produzir
    F1 | frequência | precision | recall | suporte | principais confusões | grupo Ekman
e, além disso,
    F1 do grupo Ekman | ΔF1 após agrupamento
para encontrar os CANDIDATOS NATURAIS A FUSÃO.

Como rodar
----------
Este script reaproveita as PROBABILIDADES OOF já salvas pelo baseline
(`data/out/bertimbau/cache/P_bertimbau_fold{k}.npy`).
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, f1_score

warnings.filterwarnings("ignore")

# Configuração 
ROOT = Path("")
DATA_DIR = ROOT / "data/treated"
OUT_BASE = ROOT / "data/out"
CKPT_DIR = ROOT / "data/checkpoints/bertimbau"
CACHE_DIR = OUT_BASE / "bertimbau/cache"  # onde estão os P_bertimbau_fold{k}.npy
EXPORT_DIR = OUT_BASE / "etapa2_analise"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

N_OUTER_FOLDS = 5
FIXED_THRESHOLD = 0.5
THRESH_GRID = np.linspace(0.05, 0.95, 19)   # mesma grade do baseline

EMOTION_COLS = [
    'admiration', 'amusement', 'anger', 'annoyance',
    'approval', 'caring', 'confusion', 'curiosity', 'desire',
    'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
    'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]
N_LABELS = len(EMOTION_COLS)

# Tradução PT-BR
PT = {
    'admiration':'Admiração', 'amusement':'Diversão', 'anger':'Raiva', 'annoyance':'Irritação',
    'approval':'Aprovação', 'caring':'Cuidado', 'confusion':'Confusão', 'curiosity':'Curiosidade',
    'desire':'Desejo', 'disappointment':'Decepção', 'disapproval':'Desaprovação', 'disgust':'Nojo',
    'embarrassment':'Constrangimento', 'excitement':'Entusiasmo', 'fear':'Medo', 'gratitude':'Gratidão',
    'grief':'Luto', 'joy':'Alegria', 'love':'Amor', 'nervousness':'Nervosismo', 'optimism':'Otimismo',
    'pride':'Orgulho', 'realization':'Percepção', 'relief':'Alívio', 'remorse':'Remorso',
    'sadness':'Tristeza', 'surprise':'Surpresa', 'neutral':'Neutro',
}

# Mapa de Ekman - neutral fica fora do nível Ekman
EKMAN_MAP = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "joy", "amusement", "approval", "excitement", "gratitude",
        "love", "optimism", "relief", "pride", "admiration",
        "desire", "caring"
    ],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}
EMO_TO_EKMAN = {e: g for g, emos in EKMAN_MAP.items() for e in emos}
EKMAN_COLORS = {"anger":"#e74c3c", "disgust":"#8e44ad", "fear":"#e67e22",
                "joy":"#f1c40f", "sadness":"#3498db", "surprise":"#1abc9c", "neutral":"#95a5a6"}

# labels dos folds + probabilidades OOF em cache
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
            raise FileNotFoundError(
                f"Cache OOF não encontrado: {p_path}\n"
                f"Rode o baseline_BERT.ipynb primeiro (ele salva esse .npy), "
                f"ou use o bloco FALLBACK ao final deste arquivo."
            )
        P[k] = np.load(p_path)
        assert P[k].shape == (Y[k].shape[0], N_LABELS), \
            f"Shape inconsistente no fold {k}: P={P[k].shape} Y={Y[k].shape}"
    return Y, P


# Limiares Calibrados OOF - calibra em folds != k
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
    """Monta Y_true_all, Y_pred_all(@limiar) e Y_pred_fix(@0.5) empilhando
    cada fold com o limiar calibrado nos outros folds."""
    thr_store = {}
    Yt, Yp_lim, Yp_fix, Pall = [], [], [], []
    for k in range(1, N_OUTER_FOLDS + 1):
        others = [j for j in range(1, N_OUTER_FOLDS + 1) if j != k]
        P_cal = np.vstack([P[j] for j in others])
        Y_cal = np.vstack([Y[j] for j in others])
        thr = calibrate_thresholds(P_cal, Y_cal)
        thr_store[k] = thr
        Yt.append(Y[k]); Pall.append(P[k])
        Yp_lim.append((P[k] >= thr[None, :]).astype(int))
        Yp_fix.append((P[k] >= FIXED_THRESHOLD).astype(int))
    thr_mean = np.vstack([thr_store[k] for k in thr_store]).mean(0)
    return (np.vstack(Yt), np.vstack(Yp_lim), np.vstack(Yp_fix),
            np.vstack(Pall), thr_mean)

# Tabela por emoção
def per_emotion_table(Y_true, Y_pred, thr_mean):
    prec, rec, f1, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average=None, zero_division=0, labels=range(N_LABELS))
    support = Y_true.sum(0)
    N = Y_true.shape[0]
    rows = []
    for j, emo in enumerate(EMOTION_COLS):
        rows.append({
            "emocao": emo,
            "emocao_pt": PT[emo],
            "grupo_ekman": EMO_TO_EKMAN.get(emo, "—"),
            "suporte": int(support[j]),
            "frequencia": support[j] / N,
            "precision": prec[j],
            "recall": rec[j],
            "f1": f1[j],
            "limiar": thr_mean[j],
        })
    df = pd.DataFrame(rows).sort_values("f1").reset_index(drop=True)
    return df

# CONFUSÃO MULTI-LABEL
#    Substituição: quando erra a emoção i (true_i=1 e pred_i=0),
#    o que o modelo previu no lugar?  M[i,j] = #{true_i=1, pred_i=0, pred_j=1}
#    Co-ocorrência: C[i,j] = #{true_i=1 & pred_j=1}
def confusion_matrices(Y_true, Y_pred):
    M = np.zeros((N_LABELS, N_LABELS), dtype=int)  # substituição (nos erros)
    C = np.zeros((N_LABELS, N_LABELS), dtype=int)  # co-ocorrência bruta
    for i in range(N_LABELS):
        miss_i = (Y_true[:, i] == 1) & (Y_pred[:, i] == 0)   # falsos negativos de i
        true_i = (Y_true[:, i] == 1)
        for j in range(N_LABELS):
            if i == j:
                continue
            M[i, j] = int((miss_i & (Y_pred[:, j] == 1)).sum())
            C[i, j] = int((true_i & (Y_pred[:, j] == 1)).sum())
    return M, C

def top_confusions(M, support, k=3):
    """Top-k confusões por emoção, normalizadas pelo suporte, marcando se o
    alvo pertence ao MESMO grupo de Ekman"""
    rows = []
    for i, emo in enumerate(EMOTION_COLS):
        supp = max(int(support[i]), 1)
        order = np.argsort(-M[i])
        tops = []
        for j in order[:k]:
            if M[i, j] == 0:
                break
            tgt = EMOTION_COLS[j]
            same = EMO_TO_EKMAN.get(emo) == EMO_TO_EKMAN.get(tgt) and emo != 'neutral'
            tops.append(f"{tgt}({M[i,j]}/{supp}={M[i,j]/supp:.2f}{'*' if same else ''})")
        rows.append({
            "emocao": emo,
            "grupo_ekman": EMO_TO_EKMAN.get(emo, "—"),
            "principais_confusoes": " ; ".join(tops) if tops else "—",
            "frac_erro_no_mesmo_grupo": _same_group_miss_fraction(M, support, i),
        })
    return pd.DataFrame(rows)

def _same_group_miss_fraction(M, support, i):
    """Dos erros de i redirecionados para outras emoções, que fração cai no
    mesmo grupo de Ekman? (sinal de 'erro estruturado')"""
    emo = EMOTION_COLS[i]
    grp = EMO_TO_EKMAN.get(emo)
    if grp is None:
        return np.nan
    total = M[i].sum()
    if total == 0:
        return 0.0
    same = sum(M[i, j] for j in range(N_LABELS)
               if j != i and EMO_TO_EKMAN.get(EMOTION_COLS[j]) == grp)
    return same / total


# Grupo Ekman: F1 do grupo (OR) e deltaF1 após agrupamento
def merged_labels(Y, emos):
    idx = [EMOTION_COLS.index(e) for e in emos]
    return Y[:, idx].max(axis=1)

def ekman_group_analysis(Y_true, Y_pred, f1_fine):
    """Para cada grupo: F1 do grupo (rótulo único via OR), média do F1 dos
    membros, e deltaF1 = F1_grupo − média_membros"""
    rows = []
    for grp, emos in EKMAN_MAP.items():
        emos = [e for e in emos if e in EMOTION_COLS]
        yt = merged_labels(Y_true, emos)
        yp = merged_labels(Y_pred, emos)
        f1_grp = f1_score(yt, yp, zero_division=0)
        mean_members = float(np.mean([f1_fine[e] for e in emos]))
        rows.append({
            "grupo_ekman": grp,
            "n_emocoes": len(emos),
            "emocoes": ", ".join(emos),
            "f1_medio_membros": mean_members,
            "f1_grupo_OR": f1_grp,
            "deltaF1_grupo": f1_grp - mean_members,
            "suporte_grupo": int(yt.sum()),
        })
    return pd.DataFrame(rows).sort_values("deltaF1_grupo", ascending=False).reset_index(drop=True)

def macro_f1_fine_vs_ekman(Y_true, Y_pred):
    """Macro-F1 fino (27 emoções, sem neutral) vs macro-F1 no nível Ekman (6 grupos).
    É análogo ao 0,46 -> 0,64 do artigo original do GoEmotions"""
    mapped = [e for e in EMOTION_COLS if e in EMO_TO_EKMAN]
    idx = [EMOTION_COLS.index(e) for e in mapped]
    f1_fine = f1_score(Y_true[:, idx], Y_pred[:, idx], average="macro", zero_division=0)

    groups = list(EKMAN_MAP.keys())
    YT = np.column_stack([merged_labels(Y_true, EKMAN_MAP[g]) for g in groups])
    YP = np.column_stack([merged_labels(Y_pred, EKMAN_MAP[g]) for g in groups])
    f1_ek = f1_score(YT, YP, average="macro", zero_division=0)
    return f1_fine, f1_ek, f1_ek - f1_fine

# Fusão dirigida por confusão (só funde o subconjunto ruim)
#    Simula fundir um SUBCONJUNTO S de emoções em 1 rótulo (OR) e mede o
#    deltaF1-macro global sobre as emoções mapeadas (sem neutral)
def simulate_subset_merge(Y_true, Y_pred, subset):
    mapped = [e for e in EMOTION_COLS if e in EMO_TO_EKMAN]
    idx_map = [EMOTION_COLS.index(e) for e in mapped]
    keep = [e for e in mapped if e not in set(subset)]
    idx_keep = [EMOTION_COLS.index(e) for e in keep]

    f1_orig = f1_score(Y_true[:, idx_map], Y_pred[:, idx_map],
                       average="macro", zero_division=0)
    yt_m = merged_labels(Y_true, subset)[:, None]
    yp_m = merged_labels(Y_pred, subset)[:, None]
    YT = np.hstack([Y_true[:, idx_keep], yt_m])
    YP = np.hstack([Y_pred[:, idx_keep], yp_m])
    f1_fused = f1_score(YT, YP, average="macro", zero_division=0)
    return {
        "subset": ", ".join(subset),
        "n": len(subset),
        "f1_rotulo_fundido": f1_score(yt_m.ravel(), yp_m.ravel(), zero_division=0),
        "f1_macro_sem_fusao": f1_orig,
        "f1_macro_com_fusao": f1_fused,
        "deltaF1_global": f1_fused - f1_orig,
    }

def fusion_candidate_score(per_emo, conf_df):
    """Score transparente de prioridade de fusão por emoção, combinando os
    3 critérios da literatura que são computáveis aqui:
      (1) F1 baixo  (2) frequência baixa
      (3) fração alta de erro caindo no mesmo grupo Ekman (erro estruturado)
    OBS: concordância entre anotadores NÃO entra aqui (possivel vazamento?)
    """
    m = per_emo.merge(conf_df[["emocao", "frac_erro_no_mesmo_grupo",
                               "principais_confusoes"]], on="emocao", how="left")
    m["z_baixo_f1"] = (m["f1"].max() - m["f1"]) / (m["f1"].std() + 1e-9)
    m["z_baixa_freq"] = (np.log10(m["frequencia"].max() + 1e-9)
                         - np.log10(m["frequencia"] + 1e-9))
    m["z_baixa_freq"] /= (m["z_baixa_freq"].std() + 1e-9)
    m["prioridade_fusao"] = (
        1.0 * m["z_baixo_f1"]
        + 0.5 * m["z_baixa_freq"]
        + 2.0 * m["frac_erro_no_mesmo_grupo"].fillna(0)
    )
    return m.sort_values("prioridade_fusao", ascending=False).reset_index(drop=True)

# Curva de Granularidade (Experimento, n_emoção: 28 -> 26 -> ... -> 7)
#    Recebe uma lista de "níveis", cada nível = lista de grupos-de-fusão
def granularity_point(Y_true, Y_pred, merge_groups):
    """merge_groups: lista de listas; cada sublista vira 1 rótulo (OR).
    Emoções não citadas permanecem individuais. Neutral é ignorado"""
    mapped = [e for e in EMOTION_COLS if e in EMO_TO_EKMAN]
    merged_flat = {e for g in merge_groups for e in g}
    singles = [e for e in mapped if e not in merged_flat]
    cols_T, cols_P = [], []
    for e in singles:
        j = EMOTION_COLS.index(e)
        cols_T.append(Y_true[:, j]); cols_P.append(Y_pred[:, j])
    for g in merge_groups:
        cols_T.append(merged_labels(Y_true, g)); cols_P.append(merged_labels(Y_pred, g))
    YT = np.column_stack(cols_T); YP = np.column_stack(cols_P)
    return {
        "n_classes": len(singles) + len(merge_groups),
        "macro_f1" : f1_score(YT, YP, average="macro", zero_division=0),
        "micro_f1" : f1_score(YT, YP, average="micro", zero_division=0),
    }

# Gráficos
def plot_f1_by_group(per_emo, path):
    d = per_emo.sort_values("f1")
    colors = [EKMAN_COLORS.get(g, "#bbb") for g in d["grupo_ekman"]]
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(d["emocao"], d["f1"], color=colors, alpha=0.9)
    ax.set_xlabel("F1 (OOF, BERTimbau @limiar)"); ax.set_xlim(0, 1)
    ax.set_title("F1 por emoção — cor = grupo de Ekman")
    handles = [plt.Rectangle((0,0),1,1,color=EKMAN_COLORS[g]) for g in EKMAN_COLORS if g!="neutral"]
    ax.legend(handles, [g for g in EKMAN_COLORS if g!="neutral"], loc="lower right", fontsize=8)
    ax.grid(axis="x", ls=":", alpha=0.4); plt.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)

def plot_confusion_heatmap(M, support, path):
    Mn = M / np.maximum(support[:, None], 1) # normaliza pela linha (suporte de i)
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(Mn, cmap="magma_r", aspect="auto")
    ax.set_xticks(range(N_LABELS)); ax.set_xticklabels(EMOTION_COLS, rotation=90, fontsize=7)
    ax.set_yticks(range(N_LABELS)); ax.set_yticklabels(EMOTION_COLS, fontsize=7)
    ax.set_xlabel("previsto no lugar (pred_j=1)"); ax.set_ylabel("emoção errada (true_i=1, pred_i=0)")
    ax.set_title("Confusão multi-label normalizada — substituição nos erros")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); plt.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)

def main():
    print("Carregando folds + probabilidades OOF em cache…")
    Y, P = load_folds()
    Y_true, Y_pred, Y_fix, P_all, thr_mean = build_oof_predictions(Y, P)
    N = Y_true.shape[0]
    print(f"  N amostras OOF = {N} | labels = {N_LABELS}")

    # tabela por emoção
    per_emo = per_emotion_table(Y_true, Y_pred, thr_mean)
    f1_fine = dict(zip(per_emo["emocao"], per_emo["f1"]))

    # confusões
    M, C = confusion_matrices(Y_true, Y_pred)
    conf_df = top_confusions(M, Y_true.sum(0), k=3)

    # tabela final "por emoção" já com confusões
    tabela = per_emo.merge(conf_df[["emocao", "principais_confusoes",
                                    "frac_erro_no_mesmo_grupo"]], on="emocao")
    tabela = tabela[["emocao", "emocao_pt", "grupo_ekman", "suporte", "frequencia",
                     "precision", "recall", "f1", "limiar",
                     "principais_confusoes", "frac_erro_no_mesmo_grupo"]]

    # grupos Ekman
    ekman_df = ekman_group_analysis(Y_true, Y_pred, f1_fine)
    f1f, f1e, dfe = macro_f1_fine_vs_ekman(Y_true, Y_pred)

    # fusão dirigida por confusão (subconjuntos candidatos)
    #    monta automaticamente, por grupo, o subconjunto de F1 < mediana do grupo
    candidatos_subset = []
    for grp, emos in EKMAN_MAP.items():
        emos = [e for e in emos if e in EMOTION_COLS]
        if len(emos) < 2:
            continue
        f1s = {e: f1_fine[e] for e in emos}
        med = np.median(list(f1s.values()))
        ruins = [e for e in emos if f1s[e] <= med]
        if len(ruins) >= 2:
            candidatos_subset.append((grp, ruins))
    subset_rows = []
    for grp, S in candidatos_subset:
        r = simulate_subset_merge(Y_true, Y_pred, S); r["grupo_ekman"] = grp
        subset_rows.append(r)
    # também os grupos Ekman inteiros, para comparação
    for grp, emos in EKMAN_MAP.items():
        emos = [e for e in emos if e in EMOTION_COLS]
        if len(emos) >= 2:
            r = simulate_subset_merge(Y_true, Y_pred, emos)
            r["grupo_ekman"] = grp + " (INTEIRO)"; subset_rows.append(r)
    subset_df = pd.DataFrame(subset_rows).sort_values("deltaF1_global", ascending=False)

    # score de prioridade de fusão por emoção
    prio = fusion_candidate_score(per_emo, conf_df)

    # curva de granularidade (exemplo com os subconjuntos que ajudam)
    niveis = []
    ajudam = [S for (grp, S), row in zip(candidatos_subset, subset_rows)
              if row["deltaF1_global"] > 0]
    niveis.append(("28 (original)", []))
    acc = []
    for S in ajudam:
        acc = acc + [S]
        niveis.append((f"{28 - sum(len(s)-1 for s in acc)}", list(acc)))
    niveis.append(("7 (Ekman)", [EKMAN_MAP[g] for g in EKMAN_MAP]))
    curva_rows = []
    for nome, mg in niveis:
        pt = granularity_point(Y_true, Y_pred, mg)
        curva_rows.append({"nivel": nome, **pt})
    curva_df = pd.DataFrame(curva_rows)

    tabela.to_csv(EXPORT_DIR / "etapa2_por_emocao.csv", index=False)
    ekman_df.to_csv(EXPORT_DIR / "etapa2_grupos_ekman.csv", index=False)
    subset_df.to_csv(EXPORT_DIR / "etapa2_fusoes_candidatas.csv", index=False)
    prio.to_csv(EXPORT_DIR / "etapa2_prioridade_fusao.csv", index=False)
    curva_df.to_csv(EXPORT_DIR / "etapa2_curva_granularidade.csv", index=False)
    pd.DataFrame(M, index=EMOTION_COLS, columns=EMOTION_COLS)\
        .to_csv(EXPORT_DIR / "etapa2_confusao_substituicao.csv")
    pd.DataFrame(C, index=EMOTION_COLS, columns=EMOTION_COLS)\
        .to_csv(EXPORT_DIR / "etapa2_coocorrencia.csv")

    with pd.ExcelWriter(EXPORT_DIR / "etapa2_analise.xlsx") as xw:
        tabela.round(4).to_excel(xw, sheet_name="por_emocao", index=False)
        ekman_df.round(4).to_excel(xw, sheet_name="grupos_ekman", index=False)
        subset_df.round(4).to_excel(xw, sheet_name="fusoes_candidatas", index=False)
        prio.round(4).to_excel(xw, sheet_name="prioridade_fusao", index=False)
        curva_df.round(4).to_excel(xw, sheet_name="curva_granularidade", index=False)

    plot_f1_by_group(per_emo, EXPORT_DIR / "etapa2_f1_por_grupo.png")
    plot_confusion_heatmap(M, Y_true.sum(0), EXPORT_DIR / "etapa2_confusao_heatmap.png")

    print("\n" + "="*70)
    print("MACRO-F1  fino (27, sem neutral) = %.4f  |  Ekman (6) = %.4f  |  Δ = %+.4f"
          % (f1f, f1e, dfe))
    print("="*70)
    print("\nTOP CANDIDATOS A FUSÃO (subconjuntos que MAIS aumentam o F1-macro global):")
    print(subset_df[subset_df["deltaF1_global"] > 0]
          [["grupo_ekman","subset","f1_rotulo_fundido","deltaF1_global"]]
          .round(4).to_string(index=False))
    print("\nGrupos Ekman INTEIROS que PIORAM ao fundir (não fundir cegamente):")
    print(subset_df[(subset_df["grupo_ekman"].str.contains("INTEIRO")) &
                    (subset_df["deltaF1_global"] < 0)]
          [["grupo_ekman","deltaF1_global"]].round(4).to_string(index=False))
    print("\nCurva de granularidade:")
    print(curva_df.round(4).to_string(index=False))
    print(f"\n✓ Tudo exportado em: {EXPORT_DIR.resolve()}")
    print("  - etapa2_analise.xlsx (todas as abas)")
    print("  - CSVs individuais + 2 PNGs (F1 por grupo, heatmap de confusão)")

if __name__ == "__main__":
    main()