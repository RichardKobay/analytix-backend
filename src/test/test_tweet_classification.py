from transformers import AutoModelForSequenceClassification
from transformers import XLMRobertaTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

def preprocess(tweet: str) -> str:
    new_text = []
    for t in tweet.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_data_tweets():
    pass

def process_data(tweet: str):
    MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    CACHE_DIR = ".cache/"

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL, cache_dir=CACHE_DIR)
    config = AutoConfig.from_pretrained(MODEL, cache_dir=CACHE_DIR)
    
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, cache_dir=CACHE_DIR)
    model.save_pretrained(CACHE_DIR)
    
    
    text = tweet
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    
    analysis = {}
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        score = scores[ranking[i]]
        analysis[label] = np.round(float(score), 4)
    
    return analysis

if __name__ == "__main__":
    result = process_data("I like cats so bad that I would have thousands of them")
    print(result)
