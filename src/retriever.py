import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, TEXT_COL, EMOTION_COLS

def build_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

def encode_texts(embedder, texts):
    return embedder.encode(
        [str(t) for t in texts],
        batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )

class FewShotRetriever:
    def __init__(self, pool_df, pool_emb, embedder, text_col=TEXT_COL):
        self.pool_df  = pool_df.reset_index(drop=True)
        self.pool_emb = pool_emb # (N_pool, D) normalizado
        self.embedder = embedder
        self.text_col = text_col
        self.texts    = self.pool_df[text_col].fillna("").astype(str).tolist()
        self._present_cols = [c for c in EMOTION_COLS if c in self.pool_df.columns]

    def _gold_labels(self, row):
        labels = [c for c in self._present_cols if int(row[c]) == 1]
        return labels if labels else ["neutral"]

    def search(self, query, top_k):
        q = self.embedder.encode(
            [str(query)], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        sims = self.pool_emb @ q # cosseno (normalizados)
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])] # ordena decrescente
        results = []
        for i in idx:
            row = self.pool_df.iloc[i]
            results.append({
                "score":  float(sims[i]),
                "text":   self.texts[i],
                "labels": self._gold_labels(row),
            })
        return results