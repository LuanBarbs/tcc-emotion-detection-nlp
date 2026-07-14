from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-9B"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# "local" -> carrega o Qwen neste processo (experimento standalone)
# "attached" -> usa um modelo já residente injetado via set_shared_model()
MODEL_SOURCE = "local"

N_OUTER_FOLDS = 5
RANDOM_STATE = 42
THRESHOLD = 0.5 # binariza as confianças do LLM (mesmo do BERT)

N_FEWSHOT = 8 # número de demonstrações recuperadas por exemplo

MAX_NEW_TOKENS = 512
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9

TEXT_COL = "CLEAN_TEXT"
EMOTION_COLS = [
    'admiration', 'amusement', 'anger', 'annoyance',
    'approval', 'caring', 'confusion', 'curiosity', 'desire',
    'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
    'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]
N_LABELS = len(EMOTION_COLS)

DATA_DIR      = Path("data/treated")
OUT_DIR       = Path("data/out/qwen_fewshot")
EMB_CACHE_DIR = Path("data/cache/fold_embeddings")
EXP_LOG_ROOT  = Path("logs/experiments/exp01_qwen")