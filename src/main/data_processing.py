import json
import logging

import numpy as np
import pandas as pd
import tweepy
from dotenv import dotenv_values
from scipy.special import softmax
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer,
)
from transformers import pipeline
import torch


def process_data(username: str, start_date: str, end_date: str) -> dict:
    """Will return the data of the username (all tweets with it's sentiment analysis)

    Args:
        `username` (str): the twitter username
        `start_date` (str): the timestamp of the beginning of scanning tweets (i.e. 2024-01-01T00:00:00Z)
        `end_date` (str): The timestamp of the end of scanning tweets (i.e. 2024-12-31T23:59:59Z)

    Returns:
        `dict`: A dictionary with all the information about the tweets
    """
    # df = get_user_tweets(username, start_date, end_date)
    df = pd.read_csv("data/processed/tweets_processed_copy.csv")
    df = add_sentiment_analysis(df)
    json_df = df.to_json()
    return json.loads(json_df)


def get_user_tweets(username: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get a user tweets from a start date to an end date and
    returns a dataframe with all the tweets

    Args:
        `username` (str): the twitter username
        `start_date` (str): the timestamp of the beginning of scanning tweets (i.e. 2024-01-01T00:00:00Z)
        `end_date` (str): The timestamp of the end of scanning tweets (i.e. 2024-12-31T23:59:59Z)

    Returns:
        pd.DataFrame: The dataframe with all the tweets and it's date
    """
    config = dotenv_values(".env")
    bearer_token = config["TWITTER_BEARER_TOKEN"]
    client = tweepy.Client(bearer_token=bearer_token)

    user = client.get_user(username=username)
    user_id = user.data.id

    tweets = client.get_users_tweets(
        id=user_id,
        start_time=start_date,
        end_time=end_date,
        tweet_fields=["created_at", "text"],
        max_results=100,
    )

    attributes_container = [
        [tweet["text"], tweet["created_at"]] for tweet in tweets.data
    ]

    columns = ["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"]

    return pd.DataFrame(attributes_container, columns=columns)


def add_sentiment_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the tweets dataframe and adds all the sentiment analysis

    Args:
        data (pd.DataFrame): The dataframe of the tweets

    Returns:
        pd.DataFrame: The dataframe with the tweets and all the sentiment analysis
    """
    df = xml_roberta_base_sentiment(data)
    df = distilbert_sentiment(df)
    return df

def distilbert_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sentiments = []
    for tweet in df["tweet"]:
        sentiment = process_data_with_distilbert(tweet)
        sentiments.append(sentiment)
    
    df["distilbert_sentiment"] = sentiments
    return df

def xml_roberta_base_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sentiments = []
    for tweet in df["tweet"]:
        sentiment = process_data_with_roberta(tweet)
        sentiments.append(sentiment)

    df["sentiment"] = sentiments
    return df


def preprocess_tweet(tweet: str) -> str:
    new_text = []
    for t in tweet.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def process_data_with_roberta(tweet: str) -> str:
    MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    CACHE_DIR = ".cache/"

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL, cache_dir=CACHE_DIR)
    config = AutoConfig.from_pretrained(MODEL, cache_dir=CACHE_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, cache_dir=CACHE_DIR
    )
    model.save_pretrained(CACHE_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    text = tweet
    text = preprocess_tweet(text)
    encoded_input = tokenizer(text, return_tensors="pt").to(device)
    output = model(**encoded_input)
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    highest_label = config.id2label[ranking[0]]
    return highest_label

def process_data_with_distilbert(tweet: str) -> str:
    classifier = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion"
    )

    prediction = classifier(tweet)
    
    highest_score = max(prediction, key=lambda x: x['score'])
    return highest_score['label']


if __name__ == "__main__":
    process_data("", "", "")
