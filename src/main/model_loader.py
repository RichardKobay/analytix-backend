from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer,
    pipeline
)
import torch

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
CACHE_DIR = ".cache/"

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL, cache_dir=CACHE_DIR)
config = AutoConfig.from_pretrained(MODEL, cache_dir=CACHE_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, cache_dir=CACHE_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

distilbert_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)
