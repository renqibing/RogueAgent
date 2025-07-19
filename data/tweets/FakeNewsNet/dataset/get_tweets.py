import pandas as pd
import json

# num_real = 100
# num_fake = 100

# all_real_tweets = pd.read_csv("./gossipcop_real.csv")
# all_real_titles = all_real_tweets["title"].tolist()[:num_real]
all_real_tweets = pd.read_csv("./politifact_real.csv")
all_real_titles = all_real_tweets["title"].tolist()
# print(type(all_titles))
print(len(all_real_titles))
with open("real_tweets_politics.json", "w") as file:
    json.dump(all_real_titles, file)

# all_fake_tweets = pd.read_csv("./gossipcop_fake.csv")
# all_fake_titles = all_fake_tweets["title"].tolist()[:num_fake]
all_fake_tweets = pd.read_csv("./politifact_fake.csv")
all_fake_titles = all_fake_tweets["title"].tolist()
# print(type(all_titles))
print(len(all_fake_titles))
with open("fake_tweets_politics.json", "w") as file:
    json.dump(all_fake_titles, file)
