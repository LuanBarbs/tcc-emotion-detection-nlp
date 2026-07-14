import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
)
from config import (
    MODEL_ID, MAX_NEW_TOKENS, GEN_TEMPERATURE, GEN_TOP_P,
)

# Registro global: o chatbot publica aqui o modelo já carregado em VRAM.
_SHARED = {"model": None, "tokenizer": None}

def set_shared_model(model, tokenizer):
    _SHARED["model"] = model
    _SHARED["tokenizer"] = tokenizer

def has_shared_model():
    return _SHARED["model"] is not None and _SHARED["tokenizer"] is not None

class EmotionModel:
    def chat(self, system_prompt: str, user_prompt: str, sample: bool = False) -> str:
        raise NotImplementedError

class QwenModel(EmotionModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def chat(self, system_prompt, user_prompt, sample=False):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if sample:
            gen_kwargs.update(do_sample=True, temperature=GEN_TEMPERATURE,
                              top_p=GEN_TOP_P, repetition_penalty=1.4)
        else:
            gen_kwargs.update(do_sample=False, repetition_penalty=1.2)

        out = self.model.generate(**inputs, **gen_kwargs)
        # Decodifica SOMENTE o que foi gerado após o prompt.
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

def load_local_qwen() -> QwenModel:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda",
        quantization_config=bnb, torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    return QwenModel(model, tok)

def get_model(source: str = "local") -> EmotionModel:
    if source == "attached":
        if not has_shared_model():
            raise RuntimeError(
                "MODEL_SOURCE='attached', mas nenhum modelo foi publicado. "
                "Chame set_shared_model(model, tokenizer) antes de get_model()."
            )
        return QwenModel(_SHARED["model"], _SHARED["tokenizer"])
    return load_local_qwen()