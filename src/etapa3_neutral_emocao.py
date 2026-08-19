"""
ETAPA 3 — Analise neutral + emocao

Detecta automaticamente a FONTE DOS VOTOS:
  (A) colunas de contagem `<emo>_votes`                     -> analise completa
  (B) as proprias colunas de emocao com valores > 1 (votos) -> analise completa
  (C) so rotulo binario (0/1) + n_annotators                -> modo degradado:
        itens 1,2,4,5 em nivel de ROTULO; item 3 (Fleiss-k) fica indisponivel
        (exige contagem de votos) e e sinalizado.

Rodar:  python etapa3_neutral_emocao.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# CONFIG
ROOT = Path("")
DATA_DIR = ROOT / "data/treated"
OUT_BASE = ROOT / "data/out"
ETAPA2_DIR = OUT_BASE / "etapa2_analise"
EXPORT_DIR = OUT_BASE / "etapa3_neutral"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

N_OUTER_FOLDS = 5
N_ANN_COL = "n_annotators"
NEUTRAL_COL = "neutral"

EMOTION_COLS = [
    'admiration', 'amusement', 'anger', 'annoyance',
    'approval', 'caring', 'confusion', 'curiosity', 'desire',
    'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
    'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]
EMO_NO_NEUTRAL = [e for e in EMOTION_COLS if e != NEUTRAL_COL]

EKMAN_MAP = {
    "anger"   : ["anger", "annoyance", "disapproval"],
    "disgust" : ["disgust"],
    "fear"    : ["fear", "nervousness"],
    "joy"     : ["joy", "amusement", "approval", "excitement", "gratitude",
                 "love", "optimism", "relief", "pride", "admiration",
                 "desire", "caring"],
    "sadness" : ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}
EMO_TO_EKMAN = {e: g for g, emos in EKMAN_MAP.items() for e in emos}

# le os folds e resolve a fonte dos votos
def load_all_folds():
    parts = []
    for k in range(1, N_OUTER_FOLDS + 1):
        df = pd.read_parquet(DATA_DIR / f"fold_{k}" / "data.parquet")
        df["__fold__"] = k
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if N_ANN_COL not in df.columns:
        raise ValueError(f"Coluna '{N_ANN_COL}' nao encontrada nos folds.")
    return df

def resolve_votes(df):
    """Retorna (votes_df|None, present_df, n_ann, modo).
       present_df: N x 28 booleano de PRESENCA (rotulo gold).
       votes_df  : N x 28 int de contagem de votos, ou None se indisponivel."""
    n_ann = df[N_ANN_COL].values.astype(int)

    # (A) colunas explicitas <emo>_votes ?
    vote_cols = {e: f"{e}_votes" for e in EMOTION_COLS if f"{e}_votes" in df.columns}
    if len(vote_cols) == len(EMOTION_COLS):
        votes = df[[vote_cols[e] for e in EMOTION_COLS]].copy()
        votes.columns = EMOTION_COLS
        if all(e in df.columns for e in EMOTION_COLS):
            present = df[EMOTION_COLS] > 0
        else:
            present = votes > 0
        return votes.astype(int), present.astype(bool), n_ann, "A: colunas *_votes"

    # (B) as proprias colunas de emocao sao contagens (algum valor > 1) ?
    if all(e in df.columns for e in EMOTION_COLS):
        sub = df[EMOTION_COLS]
        if (sub.values > 1).any():
            votes = sub.astype(int)
            present = (sub > 0)
            return votes, present.astype(bool), n_ann, "B: colunas de emocao = votos"
        # (C) binario puro
        present = (sub > 0)
        return None, present.astype(bool), n_ann, "C: binario (sem votos)"

    raise ValueError("Nao encontrei nem colunas de emocao nem colunas *_votes nos folds.")

# FREQUENCIA DE neutral + emocao
def situation_table(present):
    n_emo = present[EMO_NO_NEUTRAL].sum(axis=1).values
    has_neu = present[NEUTRAL_COL].values
    has_emo = n_emo > 0
    cat = np.full(len(present), "vazio", dtype=object)
    cat[has_neu & ~has_emo] = "so neutral"
    cat[~has_neu & has_emo & (n_emo == 1)] = "so emocao (1)"
    cat[~has_neu & has_emo & (n_emo >= 2)] = "so emocoes (2+)"
    cat[has_neu & has_emo] = "neutral + emocao"

    tab = (pd.Series(cat).value_counts().rename_axis("situacao")
           .reset_index(name="qtd"))
    tab["pct"] = tab["qtd"] / len(present)

    coex = has_neu & has_emo
    coex_by_n = (pd.Series(n_emo[coex]).value_counts()
                 .rename_axis("n_emocoes_com_neutral")
                 .reset_index(name="qtd").sort_values("n_emocoes_com_neutral"))
    return tab, coex_by_n, cat

# PROPORCAO POR EMOCAO - P(neutral | emocao)
def per_emotion_neutral(present, votes, n_ann):
    neu_present = present[NEUTRAL_COL].values
    rows = []
    for emo in EMO_NO_NEUTRAL:
        p = present[emo].values
        supp = int(p.sum())
        if supp == 0:
            continue
        with_neu = p & neu_present
        n_with = int(with_neu.sum())
        row = {
            "emocao": emo,
            "grupo_ekman": EMO_TO_EKMAN.get(emo, "—"),
            "suporte": supp,
            "n_com_neutral": n_with,
            "prop_neutral_dado_emocao": n_with / supp,
        }
        if votes is not None:
            v   = votes[emo].values
            neu = votes[NEUTRAL_COL].values
            row["vote_ratio_medio"]  = float(np.mean(v[p] / n_ann[p]))
            row["pct_minoria_1voto"] = float(np.mean(v[p] == 1))
            denom = v[with_neu] + neu[with_neu]
            share = np.divide(neu[with_neu], denom,
                              out=np.zeros_like(denom, float), where=denom > 0)
            row["neutral_share_medio"] = float(share.mean()) if n_with else 0.0
            row["empate_neutral_ge_emocao"] = int(np.sum(
                (neu[with_neu] >= v[with_neu]) & (v[with_neu] > 0)))
        else:
            row["n_annotators_medio"] = float(np.mean(n_ann[p]))
            row["n_annotators_medio_coex"] = (float(np.mean(n_ann[with_neu]))
                                              if n_with else np.nan)
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values("prop_neutral_dado_emocao", ascending=False)
            .reset_index(drop=True))

# CONCORDANCIA - Fleiss-k por emocao (so se houver votos)
def fleiss_kappa_binary(votes_present, n_raters):
    v = np.asarray(votes_present, float); n = np.asarray(n_raters, float)
    ok = n >= 2; v, n = v[ok], n[ok]
    if len(v) == 0:
        return np.nan
    P_i = (v*(v-1) + (n-v)*(n-v-1)) / (n*(n-1))
    Pbar = P_i.mean()
    p_pres = v.sum()/n.sum(); Pe = p_pres**2 + (1-p_pres)**2
    return float((Pbar - Pe)/(1 - Pe)) if (1 - Pe) > 1e-12 else np.nan

def agreement_table(votes, present, n_ann):
    rows = []
    for emo in EMOTION_COLS:
        kap = (fleiss_kappa_binary(votes[emo].values, n_ann)
               if votes is not None else np.nan)
        rows.append({
            "emocao": emo,
            "grupo_ekman": EMO_TO_EKMAN.get(emo, "—"),
            "fleiss_kappa": kap,
            "suporte": int(present[emo].values.sum()),
        })
    return pd.DataFrame(rows).sort_values("fleiss_kappa").reset_index(drop=True)

# RELACAO COM F1 e CONFUSAO
def load_etapa2():
    f1 = conf = None
    p1 = ETAPA2_DIR / "etapa2_por_emocao.csv"
    if p1.exists():
        d = pd.read_csv(p1)
        cols = ["emocao", "f1"]
        if "frac_erro_no_mesmo_grupo" in d.columns:
            cols.append("frac_erro_no_mesmo_grupo")
        f1 = d[cols]
    p2 = ETAPA2_DIR / "etapa2_confusao_substituicao.csv"
    if p2.exists():
        Mdf = pd.read_csv(p2, index_col=0)
        conf = pd.DataFrame({"emocao": Mdf.index,
                             "massa_confusao": Mdf.values.sum(axis=1)})
    return f1, conf

def convergencia(per_emo_neu, agree, f1_df, conf_df):
    m = per_emo_neu.merge(agree[["emocao", "fleiss_kappa"]], on="emocao", how="left")
    if f1_df   is not None: m = m.merge(f1_df,   on="emocao", how="left")
    if conf_df is not None: m = m.merge(conf_df, on="emocao", how="left")
    return m

def corr_report(m):
    def sp(a, b):
        if a not in m.columns or b not in m.columns: return None
        d = m[[a, b]].dropna()
        if len(d) < 4: return (np.nan, np.nan, len(d))
        r, p = spearmanr(d[a], d[b]); return (r, p, len(d))
    out = {}
    for name, a, b in [
        ("neutral_coex_x_F1", "prop_neutral_dado_emocao", "f1"),
        ("fleiss_x_F1", "fleiss_kappa", "f1"),
        ("neutral_coex_x_confusao", "prop_neutral_dado_emocao", "massa_confusao"),
        ("neutral_coex_x_erro_estruturado", "prop_neutral_dado_emocao",
         "frac_erro_no_mesmo_grupo"),
    ]:
        r = sp(a, b)
        if r is not None: out[name] = r
    return out

# Gráficos
def scatter(m, xcol, ycol, path, title):
    if xcol not in m.columns or ycol not in m.columns: return
    d = m[["emocao", xcol, ycol]].dropna()
    if len(d) < 3: return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d[xcol], d[ycol], alpha=0.8)
    for _, r in d.iterrows():
        ax.annotate(r["emocao"], (r[xcol], r[ycol]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    rho, p = spearmanr(d[xcol], d[ycol])
    ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    ax.set_title(f"{title}\nSpearman rho={rho:.3f}  p={p:.4f}")
    ax.grid(ls=":", alpha=0.4); plt.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)

def bar_prop_neutral(per_emo_neu, path):
    d = per_emo_neu.sort_values("prop_neutral_dado_emocao", ascending=True)
    palette = {"anger":"#e74c3c","disgust":"#8e44ad","fear":"#e67e22","joy":"#f1c40f",
               "sadness":"#3498db","surprise":"#1abc9c"}
    colors = [palette.get(g, "#95a5a6") for g in d["grupo_ekman"]]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(d["emocao"], d["prop_neutral_dado_emocao"], color=colors, alpha=0.9)
    ax.set_xlabel("P(neutral | emocao)")
    ax.set_title("Coexistencia neutral+emocao por emocao (cor = grupo Ekman)")
    ax.grid(axis="x", ls=":", alpha=0.4); plt.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)

def main():
    print("Carregando folds...")
    df = load_all_folds()
    votes, present, n_ann, modo = resolve_votes(df)
    N = len(df)
    print(f"  {N} mensagens | fonte de votos = [{modo}]")
    print(f"  n_annotators: min={n_ann.min()} max={n_ann.max()} media={n_ann.mean():.2f}")
    if votes is None:
        print("  [aviso] Sem contagem de votos -> item (3) Fleiss-k indisponivel.")
        print("          Para habilita-lo, inclua colunas '<emo>_votes' nos folds.")

    sit, coex_by_n, cat = situation_table(present)
    print("\n(1) Frequencia de situacoes:")
    print(sit.to_string(index=False))

    per_emo_neu = per_emotion_neutral(present, votes, n_ann)

    agree = agreement_table(votes, present, n_ann)
    if votes is not None:
        kg = np.nanmean(agree.loc[agree.emocao != NEUTRAL_COL, "fleiss_kappa"])
        print(f"\n(3) Fleiss-k medio (emocoes, sem neutral) = {kg:.3f}")

    f1_df, conf_df = load_etapa2()
    if f1_df is None:
        print("\n[aviso] etapa2_por_emocao.csv nao encontrado -> itens (4)/(5) parciais. "
              "Rode a Etapa 2 antes.")
    conv = convergencia(per_emo_neu, agree, f1_df, conf_df)
    corrs = corr_report(conv)
    if corrs:
        print("\n(4/5) Correlacoes (Spearman):")
        for k, (r, p, n) in corrs.items():
            print(f"  {k:34s} rho={r:+.3f}  p={p:.4f}  (n={n})")

    conv_sorted = conv.copy()
    if "f1" in conv_sorted.columns:
        cond = ((conv_sorted["prop_neutral_dado_emocao"] >=
                 conv_sorted["prop_neutral_dado_emocao"].median()) &
                (conv_sorted["f1"] <= conv_sorted["f1"].median()))
        if conv_sorted["fleiss_kappa"].notna().any():
            cond &= (conv_sorted["fleiss_kappa"] <=
                     conv_sorted["fleiss_kappa"].median())
        conv_sorted["flag_problematica"] = cond
        conv_sorted = conv_sorted.sort_values(
            ["flag_problematica", "prop_neutral_dado_emocao"], ascending=[False, False])

    sit.to_csv(EXPORT_DIR / "etapa3_situacoes.csv", index=False)
    coex_by_n.to_csv(EXPORT_DIR / "etapa3_coex_por_n_emocoes.csv", index=False)
    per_emo_neu.to_csv(EXPORT_DIR / "etapa3_por_emocao_neutral.csv", index=False)
    agree.to_csv(EXPORT_DIR / "etapa3_concordancia.csv", index=False)
    conv_sorted.to_csv(EXPORT_DIR / "etapa3_convergencia.csv", index=False)
    with open(EXPORT_DIR / "etapa3_correlacoes.json", "w", encoding="utf-8") as f:
        json.dump({k: {"rho": v[0], "p": v[1], "n": v[2]} for k, v in corrs.items()},
                  f, indent=2, ensure_ascii=False)
    with pd.ExcelWriter(EXPORT_DIR / "etapa3_analise.xlsx") as xw:
        sit.round(4).to_excel(xw, sheet_name="situacoes", index=False)
        coex_by_n.to_excel(xw, sheet_name="coex_por_n", index=False)
        per_emo_neu.round(4).to_excel(xw, sheet_name="por_emocao_neutral", index=False)
        agree.round(4).to_excel(xw, sheet_name="concordancia", index=False)
        conv_sorted.round(4).to_excel(xw, sheet_name="convergencia", index=False)

    bar_prop_neutral(per_emo_neu, EXPORT_DIR / "etapa3_prop_neutral.png")
    scatter(conv, "prop_neutral_dado_emocao", "f1",
            EXPORT_DIR / "etapa3_neutralcoex_vs_f1.png", "Coexistencia com neutral x F1")
    scatter(conv, "fleiss_kappa", "f1",
            EXPORT_DIR / "etapa3_kappa_vs_f1.png", "Concordancia (Fleiss-k) x F1")
    scatter(conv, "prop_neutral_dado_emocao", "massa_confusao",
            EXPORT_DIR / "etapa3_neutralcoex_vs_confusao.png",
            "Coexistencia com neutral x massa de confusao")

    print(f"\nOK Exportado em {EXPORT_DIR.resolve()}  (etapa3_analise.xlsx + CSVs + JSON + PNGs)")
    if "flag_problematica" in conv_sorted.columns:
        prob = conv_sorted[conv_sorted["flag_problematica"]]["emocao"].tolist()
        extra = ', k baixo' if votes is not None else ''
        print(f"Emocoes com convergencia de sinais (neutral alto, F1 baixo{extra}): {prob}")

if __name__ == "__main__":
    main()