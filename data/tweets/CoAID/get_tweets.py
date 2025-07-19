import pandas as pd
import json

dirs = ['05-01-2020','07-01-2020','09-01-2020','11-01-2020']

all_real_titles = []
for dir in dirs:
    real_tweets = pd.read_csv(f"./{dir}/NewsRealCOVID-19.csv")
    real_titles = real_tweets["title"].tolist()
    all_real_titles.extend(real_titles)
print(len(all_real_titles))
with open("real_tweets_COVID.json", "w") as file:
    json.dump(all_real_titles, file)

all_fake_titles = []
for dir in dirs:
    fake_tweets = pd.read_csv(f"./{dir}/NewsFakeCOVID-19.csv")
    fake_titles = fake_tweets["title"].tolist()
    all_fake_titles.extend(fake_titles)
print(len(all_fake_titles))
with open("fake_tweets_COVID.json", "w") as file:
    json.dump(all_fake_titles, file)
