import re
import json
import numpy as np
from config import EMOTION_COLS, N_LABELS

_COL_IDX = {c.lower(): i for i, c in enumerate(EMOTION_COLS)}

def extract_json(text):
    if not text:
        return None
    text = re.sub(r"```json|```", "", text).strip()
    matches = re.findall(r"\{.*\}", text, re.DOTALL) # maior bloco {...}
    for cand in reversed(matches):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None

# Converte a saída do LLM em (y_pred, y_proba) de tamanho N_LABELS
def scores_to_vectors(parsed, threshold):
    if not parsed:
        return None, None
    emo = parsed.get("emotions", parsed)
    proba = np.zeros(N_LABELS, dtype=np.float32)

    if isinstance(emo, dict):
        for k, v in emo.items():
            j = _COL_IDX.get(str(k).strip().lower())
            if j is None:
                continue
            try:
                proba[j] = float(v)
            except (TypeError, ValueError):
                proba[j] = 1.0 # citou a emoção mas sem número -> presença
    elif isinstance(emo, list):
        for k in emo:
            j = _COL_IDX.get(str(k).strip().lower())
            if j is not None:
                proba[j] = 1.0
    else:
        return None, None

    proba = np.clip(proba, 0.0, 1.0)
    pred  = (proba >= threshold).astype(int)
    return pred, proba

def format_demonstrations(results):
    blocks = []
    for r in results:
        labels = ", ".join(r["labels"])
        blocks.append(f'Texto: "{r["text"]}"\nEmoções: {labels}')
    return "\n\n".join(blocks)