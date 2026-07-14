import numpy as np
from config import EMOTION_COLS, N_FEWSHOT, THRESHOLD, N_LABELS
from utils import extract_json, scores_to_vectors, format_demonstrations

class EmotionPipeline:
    def __init__(self, model, retriever, n_fewshot=N_FEWSHOT, threshold=THRESHOLD):
        self.model = model # EmotionModel (local ou attached)
        self.retriever = retriever # FewShotRetriever do fold atual
        self.n_fewshot = n_fewshot
        self.threshold = threshold

    def get_system_prompt(self):
        return (
            "Você é um classificador de emoções multirrótulo. Dado um texto, "
            "identifique TODAS as emoções presentes, escolhendo exclusivamente "
            "entre a lista fornecida. Para cada emoção presente, atribua uma "
            "confiança entre 0 e 1. Responda SOMENTE com JSON válido, sem "
            "explicações."
        )

    def get_user_prompt(self, text, demos, retry=False):
        labels_str = ", ".join(EMOTION_COLS)
        retry_note = ""
        if retry:
            retry_note = ("\nATENÇÃO: sua resposta anterior foi inválida. "
                          "Responda ESTRITAMENTE no formato JSON pedido, "
                          "sem nenhum texto fora do JSON.")
        return f"""Emoções permitidas (use exatamente estes nomes, em inglês):
{labels_str}

Exemplos rotulados semelhantes (referência):
{demos}

Classifique agora o TEXTO abaixo. Inclua no JSON apenas as emoções presentes,
cada uma com sua confiança (0 a 1). Se nenhuma emoção se aplicar, use "neutral".

TEXTO: "{text}"
{retry_note}
Formato EXATO da resposta:
{{"emotions": {{"nome_da_emocao": confianca}}}}
"""

    def run(self, text):
        results = self.retriever.search(text, self.n_fewshot)
        demos   = format_demonstrations(results)

        raw = self.model.chat(
            self.get_system_prompt(),
            self.get_user_prompt(text, demos),
            sample=False,
        )
        parsed = extract_json(raw)
        pred, proba = scores_to_vectors(parsed, self.threshold)
        retried = False

        # Retry com sampling
        if pred is None:
            raw = self.model.chat(
                self.get_system_prompt(),
                self.get_user_prompt(text, demos, retry=True),
                sample=True,
            )
            parsed = extract_json(raw)
            pred, proba = scores_to_vectors(parsed, self.threshold)
            retried = True

        # Falha total -> prediz "nada" (vetor zero)
        # linha-a-linha com Y_true
        failed = pred is None
        if failed:
            proba = np.zeros(N_LABELS, dtype=np.float32)
            pred  = np.zeros(N_LABELS, dtype=int)

        return {
            "pred": pred, "proba": proba, "parsed": parsed,
            "raw": raw, "retrieval": results,
            "retried": retried, "failed": failed,
        }