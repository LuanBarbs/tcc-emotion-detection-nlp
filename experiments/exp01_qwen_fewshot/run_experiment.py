import sys, os, json, argparse, warnings
from datetime import datetime

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    DATA_DIR, OUT_DIR, EMB_CACHE_DIR, EXP_LOG_ROOT,
    N_OUTER_FOLDS, MODEL_ID, EMBEDDING_MODEL, MODEL_SOURCE,
    N_FEWSHOT, THRESHOLD, TEXT_COL, EMOTION_COLS, N_LABELS,
)
from retriever import build_embedder, encode_texts, FewShotRetriever
from model_provider import get_model
from pipeline import EmotionPipeline
from metrics import compute_metrics

CURRENT_RUN_POINTER = EXP_LOG_ROOT / "current_run.txt"

def resolve_run_dir(reset: bool) -> str:
    EXP_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not reset and CURRENT_RUN_POINTER.exists():
        run_name = CURRENT_RUN_POINTER.read_text().strip()
        run_dir  = EXP_LOG_ROOT / run_name
        if run_dir.is_dir():
            return str(run_dir)
    run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = EXP_LOG_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CURRENT_RUN_POINTER.write_text(run_name)
    return str(run_dir)

def load_fold(fold_id):
    fold_dir = DATA_DIR / f"fold_{fold_id}"
    df  = pd.read_parquet(fold_dir / "data.parquet").reset_index(drop=True)
    return df

def load_all_folds():
    folds = {}
    for k in range(1, N_OUTER_FOLDS + 1):
        folds[k] = load_fold(k)
        print(f"[INFO] Fold {k}: {folds[k].shape[0]:>6} amostras")
    return folds

def get_fold_embeddings(embedder, folds):
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    embs = {}
    for k, df in folds.items():
        cache = EMB_CACHE_DIR / f"fold_{k}.npy"
        if cache.exists():
            embs[k] = np.load(cache)
            print(f"[INFO] Embeddings fold {k}: cache ({embs[k].shape})")
        else:
            print(f"[INFO] Codificando fold {k}...")
            embs[k] = encode_texts(embedder, df[TEXT_COL].tolist())
            np.save(cache, embs[k])
    return embs

def labels_matrix(df):
    present = [c for c in EMOTION_COLS if c in df.columns]
    M = np.zeros((len(df), N_LABELS), dtype=int)
    for j, c in enumerate(EMOTION_COLS):
        if c in present:
            M[:, j] = df[c].values.astype(int)
    return M

def load_checkpoint(run_dir):
    f = os.path.join(run_dir, "checkpoint_raw_results.jsonl")
    done = {}
    if not os.path.exists(f):
        return done
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[(rec["fold"], rec["pos"])] = rec
            except Exception:
                pass
    return done

def write_checkpoint(run_dir, record):
    f = os.path.join(run_dir, "checkpoint_raw_results.jsonl")
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush(); os.fsync(fh.fileno())

def update_progress(run_dir, current, total, start):
    elapsed = (datetime.now() - start).total_seconds()
    pct  = current / total * 100 if total else 0
    rate = current / elapsed if elapsed > 0 else 0
    eta  = (total - current) / rate if rate > 0 else 0
    with open(os.path.join(run_dir, "progress.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([
            f"Rodada         : {os.path.basename(run_dir)}",
            f"Atualizado em  : {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Progresso      : {current}/{total} ({pct:.1f}%)",
            f"Velocidade     : {rate:.2f} exemplos/s",
            f"ETA            : {int(eta//3600):02d}h{int((eta%3600)//60):02d}m",
        ]) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="Cria uma rodada nova, ignorando o checkpoint.")
    args = ap.parse_args()

    run_dir = resolve_run_dir(reset=args.reset)
    timestamp = os.path.basename(run_dir).replace("run_", "")
    if args.reset:
        ck = os.path.join(run_dir, "checkpoint_raw_results.jsonl")
        if os.path.exists(ck):
            os.remove(ck)
    print(f"[INFO] Rodada ativa: {run_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dados + embeddings por fold
    folds  = load_all_folds()
    labels = {k: labels_matrix(df) for k, df in folds.items()}
    embedder = build_embedder()
    embs   = get_fold_embeddings(embedder, folds)

    # 2. Modelo (local agora; "attached" quando plugar no chatbot)
    print(f"[INFO] Carregando modelo QWEN (source={MODEL_SOURCE})...")
    model = get_model(MODEL_SOURCE)
    print("[INFO] Modelo pronto!\n")

    # 3. Checkpoint + total de exemplos
    done  = load_checkpoint(run_dir)
    total = sum(len(df) for df in folds.values())
    start = datetime.now()
    processed = len(done)

    # 4. Loop OUTER-CV (mesma partição do baseline)
    for k in range(1, N_OUTER_FOLDS + 1):
        df_test = folds[k]

        pool_df  = pd.concat([folds[j] for j in folds if j != k], ignore_index=True)
        pool_emb = np.concatenate([embs[j] for j in folds if j != k], axis=0)
        retriever = FewShotRetriever(pool_df, pool_emb, embedder)
        pipe = EmotionPipeline(model, retriever, n_fewshot=N_FEWSHOT, threshold=THRESHOLD)

        pbar = tqdm(range(len(df_test)), desc=f"Fold {k}/{N_OUTER_FOLDS}",
                    unit="ex", dynamic_ncols=True)
        for pos in pbar:
            if (k, pos) in done:
                continue
            text   = str(df_test.iloc[pos][TEXT_COL])
            y_true = labels[k][pos].tolist()

            out = pipe.run(text)

            record = {
                "fold": k, "pos": pos,
                "y_true":  y_true,
                "y_proba": [float(x) for x in out["proba"]],
                "retried": out["retried"],
                "failed":  out["failed"],
            }
            write_checkpoint(run_dir, record)
            done[(k, pos)] = record
            processed += 1
            update_progress(run_dir, processed, total, start)
        pbar.close()

    # 5. Reconstroi métricas POR FOLD a partir dos probas salvos
    all_fold_metrics = []
    for k in range(1, N_OUTER_FOLDS + 1):
        recs = sorted([r for (kk, _), r in done.items() if kk == k],
                      key=lambda r: r["pos"])
        Y_true  = np.array([r["y_true"]  for r in recs], dtype=int)
        Y_proba = np.array([r["y_proba"] for r in recs], dtype=np.float32)
        Y_pred  = (Y_proba >= THRESHOLD).astype(int)

        m = compute_metrics(Y_true, Y_pred, Y_proba, EMOTION_COLS)
        m["fold"]  = k
        m["epoch"] = 0                       # N/A no few-shot (mantém schema)
        m["train_loss"] = float("nan")
        m["val_loss"]   = float("nan")
        m["elapsed_s"]  = float("nan")
        m["n_retried"]  = int(sum(r["retried"] for r in recs))
        m["n_failed"]   = int(sum(r["failed"]  for r in recs))
        all_fold_metrics.append(m)
        print(f"[FOLD {k}] F1-macro={m['f1_macro']:.4f} "
              f"F1-micro={m['f1_micro']:.4f} ROC-AUC={m['roc_auc_macro']:.4f} "
              f"(retried={m['n_retried']}, failed={m['n_failed']})")

    # 6. Exporta EXATAMENTE os mesmos artefatos do baseline (prefixo qwen_)
    export_results(all_fold_metrics, run_dir, timestamp)
    print(f"\n[DONE] Artefatos em: {run_dir} e {OUT_DIR}")

def export_results(all_fold_metrics, run_dir, timestamp):
    from plots import save_plots

    global_keys = [
        "fold", "epoch", "accuracy", "hamming_loss",
        "f1_macro", "f1_micro", "f1_weighted",
        "precision_macro", "recall_macro", "roc_auc_macro",
        "train_loss", "val_loss", "elapsed_s", "n_retried", "n_failed",
    ]
    rows = [{k: m.get(k) for k in global_keys} for m in all_fold_metrics]
    df_results = pd.DataFrame(rows).set_index("fold")

    f1_per_label_folds = pd.DataFrame(
        [m["f1_per_label"] for m in all_fold_metrics],
        index=[f"fold_{m['fold']}" for m in all_fold_metrics],
    )
    f1_per_label_mean = f1_per_label_folds.mean().sort_values(ascending=False)

    # CSVs + JSON (mesmos nomes/estrutura, prefixo qwen_)
    df_results.to_csv(OUT_DIR / "qwen_global_metrics.csv")
    f1_per_label_folds.to_csv(OUT_DIR / "qwen_f1_per_label.csv")

    summary = {
        "model":       f"QWEN few-shot (retrieval, k={N_FEWSHOT})",
        "model_id":    MODEL_ID,
        "embedding":   EMBEDDING_MODEL,
        "strategy":    f"Outer CV ({N_OUTER_FOLDS} folds), zero fine-tuning",
        "n_labels":    N_LABELS,
        "f1_macro_mean": round(float(df_results["f1_macro"].mean()), 4),
        "f1_macro_std":  round(float(df_results["f1_macro"].std()),  4),
        "f1_micro_mean": round(float(df_results["f1_micro"].mean()), 4),
        "f1_micro_std":  round(float(df_results["f1_micro"].std()),  4),
        "roc_auc_mean":  round(float(df_results["roc_auc_macro"].mean()), 4),
        "roc_auc_std":   round(float(df_results["roc_auc_macro"].std()),  4),
    }
    with open(OUT_DIR / "qwen_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Cópia dos artefatos dentro da rodada também
    df_results.to_csv(os.path.join(run_dir, f"qwen_global_metrics_{timestamp}.csv"))

    save_plots(df_results, f1_per_label_folds, f1_per_label_mean, OUT_DIR)
    print("\n===== SUMÁRIO QWEN =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()