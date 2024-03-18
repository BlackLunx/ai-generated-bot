from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


MAX_LEN = 1024

CHECKPOINT_PATH = "./deberta"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT_PATH, max_position_embeddings=MAX_LEN
).to("cuda")

async def check_text(text: str) -> bool:
    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to("cuda")
    logits = model(**inputs).logits.detach().cpu().numpy()
    return bool(np.argmax(np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)))