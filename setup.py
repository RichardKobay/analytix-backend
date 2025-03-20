import os
import pandas as pd

def download_test_datasets(dataset_url: str) -> None:
    dataset_url = "mmmarchetti/tweets-dataset"
    destination_dir = "data/raw"
    os.system(f"kaggle datasets download -d {dataset_url} -p {destination_dir} --unzip")

def process_datasets():
    dataset_path = "data/raw/tweets.csv"
    df = pd.read_csv(dataset_path)
    df = df[['content', 'date_time']]
    df.rename(columns={'content': 'tweet'}, inplace=True)
    df['date_time'] = pd.to_datetime(df['date_time'], format='%d/%m/%Y %H:%M').dt.strftime('%Y-%m-%d %H:%M:%S%z')
    df.to_csv("data/processed/tweets_processed.csv", index=False)
    print(df.head())

def process_dataset_for_testing():
    dataset_path = "data/raw/tweets.csv"
    df = pd.read_csv(dataset_path)
    df = df[['author', 'content', 'date_time']]
    
    usernames = df['author'].unique()
    for idx, username in enumerate(usernames, start=1):
        user_df = df[df['author'] == username].copy()
        user_df.drop(columns=['author'], inplace=True)
        user_df.rename(columns={'content': 'tweet'}, inplace=True)
        file_index = f"{idx:02d}"
        user_df.to_csv(f"data/processed/tweets_per_user/tweets_{username}_processed_{file_index}.csv", index=False)

if __name__ == "__main__":
    download_test_datasets("mmmarchetti/tweets-dataset")
    process_datasets()
    process_dataset_for_testing()
