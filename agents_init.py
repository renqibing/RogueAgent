from typing import Optional, Union, List
import re
import os
import json
from typing import Union, List, Optional
import pandas as pd
import ast
from datetime import datetime
import uuid
import random
import networkx as nx
import numpy as np

random.seed(42)


class AgentGenerator:
    def __init__(
        self,
        save_dir,
        root_dir,
        profile_dir,
        num_good,
        num_bad,
        good_type="good",
        bad_type="bad",
        net_structure="random",
        activity_level_distribution="uniform",
        debunking=False,
        suffix=None,
        good_posts_per_good_agent=1,
        bad_posts_per_bad_agent=9,
    ):
        """
        Initializes the AgentGenerator.

        Args:
            save_dir (str): Directory to save the CSV file.
            root_dir (str): Root directory for the project.
            num_good (int): Number of good agents to generate.
            num_bad (int): Number of bad agents to generate.
        """
        self.save_dir = os.path.join(root_dir, save_dir)
        self.profile_dir = os.path.join(root_dir, profile_dir)
        self.num_good = num_good
        self.num_bad = num_bad
        self.good_type = good_type
        self.bad_type = bad_type
        self.net_structure = net_structure
        self.activity_level_distribution = activity_level_distribution
        self.debunking = debunking
        self.sum = num_good + num_bad
        self.good_posts_per_good_agent = good_posts_per_good_agent
        self.bad_posts_per_bad_agent = bad_posts_per_bad_agent
        with open("./data/tweets/real_tweets_COVID.json", "r") as file:
            self.real_baseline_tweets = json.load(file)
        self.real_tweet_count = 0
        self.real_tweet_sum = len(self.real_baseline_tweets)
        with open("./data/tweets/fake_tweets_COVID.json", "r") as file:
            self.fake_baseline_tweets = json.load(file)
        self.fake_tweet_count = 0
        self.fake_tweet_sum = len(self.fake_baseline_tweets)
        # Shuffle the tweets
        random.shuffle(self.real_baseline_tweets)
        random.shuffle(self.fake_baseline_tweets)

        if self.debunking:
            self.filename = f"test_{self.sum}_{self.good_type}_{self.bad_type}_{self.net_structure}_{self.activity_level_distribution}_debunking_{suffix}.csv"
        else:
            self.filename = f"test_{self.sum}_{self.good_type}_{self.bad_type}_{self.net_structure}_{self.activity_level_distribution}_{suffix}.csv"
        # self.user_char = "You are a robot. Your task is to repost any post you receive."
        self.agents = []

    # def gen_user_profile(self):
    #     """
    #     Generates user profile attributes: name, user_char, username.
    #     """
    #     name = f"user_{uuid.uuid4().hex[:8]}"
    #     username = f"{name}_username"
    #     user_char = self.user_char
    #     return name, user_char, username

    def sample_activity_level_frequency(self):
        """
        Samples activity probability from a binary distribution.

        Returns:
            list: A list representing hourly activity probabilities.
        """
        if self.activity_level_distribution == "uniform":
            activity_level_frequency = [
                random.uniform(0, 1) for _ in range(24)
            ]  # uniform distribution
        elif self.activity_level_distribution == "bernoulli":
            activity_level_frequency = [
                random.choices([0, 0.2], weights=[0.9, 0.1])[0] for _ in range(24)
            ]
        elif self.activity_level_distribution == "multimodal":
            peak_times = [
                random.uniform(7, 9),
                random.uniform(11, 13),
                random.uniform(17, 19),
                random.uniform(22, 24),
            ]
            peak_heights = [0.7, 0.8, 0.9, 0.8]
            peak_widths = [1.0, 1.0, 1.0, 1.0]
            # B = 0.1  # baseline

            activity_level_frequency = np.zeros(24)
            for phi, A, sigma in zip(peak_times, peak_heights, peak_widths):
                dist = self.circular_distance(np.arange(24), phi)
                # gaussian function
                activity_level_frequency += A * \
                    np.exp(-0.5 * (dist / sigma) ** 2)

            activity_level_frequency = np.clip(activity_level_frequency, 0, 1)
        else:
            raise ValueError(
                f"Unknown activity level distribution: {self.activity_level_distribution}"
            )
        return [round(num, 3) for num in activity_level_frequency]

    def circular_distance(self, t, center, period=24):
        """Cauculate the distance between time t and center under given period."""
        raw_dist = np.abs(t - center)
        return np.minimum(raw_dist, period - raw_dist)

    def gen_network_structure(self, num_agents):
        """
        Generates a network structure and calculates followers and following.

        Args:
            num_agents (int): Total number of agents to create.

        Returns:
            list: List of agent dictionaries with updated network structure.
        """
        if self.net_structure == "random":
            G = nx.erdos_renyi_graph(num_agents, 0.1)
        elif self.net_structure == "scale_free":
            G = nx.barabasi_albert_graph(num_agents, m=2)
        elif self.net_structure == "high_clustering":
            G = nx.watts_strogatz_graph(num_agents, k=2, p=0.1)
        else:
            raise ValueError(
                f"Unknown network structure: {self.net_structure}")

        for i in range(num_agents):
            agent = self.agents[i]
            following_ids = list(G.neighbors(i))
            agent["following_agentid_list"] = following_ids
            agent["followers_count"] = sum(
                [1 for j in range(num_agents) if i in G.neighbors(j)]
            )
            agent["following_list"] = [self.agents[j]["user_id"]
                                       for j in following_ids]

        return self.agents

    def sample_tweets(self, user_type):
        """
        Samples a tweet from a dataset.

        Returns:
            str: A sampled tweet.
        """
        # baseline_tweets = [
        #     "The moon landing was staged by the government to win the space race.",
        #     "Vaccines are a tool for tracking the population.",
        #     "Climate change is a fabrication to push green energy agendas.",
        #     "Certain foods can completely cure cancer if consumed daily.",
        #     "5G technology weakens the immune system and causes illness.",
        #     "The earth is flat and evidence to the contrary is fabricated.",
        #     "The COVID-19 pandemic was planned years in advance.",
        #     "Chemtrails are a method for governments to control weather patterns.",
        #     "Cures for major diseases are being hidden by pharmaceutical companies to maintain profits.",
        #     "Artificial intelligence will fully replace human jobs in five years.",
        # ]

        if user_type.startswith("good"):
            while self.real_tweet_count >= self.real_tweet_sum:
                self.real_baseline_tweets += self.real_baseline_tweets
                self.real_tweet_sum = len(self.real_baseline_tweets)
            real_baseline_tweets = self.real_baseline_tweets[
                self.real_tweet_count: self.real_tweet_count + self.good_posts_per_good_agent]
            self.real_tweet_count += 1
            return real_baseline_tweets
        else:
            while self.fake_tweet_count >= self.fake_tweet_sum:
                self.fake_baseline_tweets += self.fake_baseline_tweets
                self.fake_tweet_sum = len(self.fake_baseline_tweets)
            fake_baseline_tweets = self.fake_baseline_tweets[
                self.fake_tweet_count: self.fake_tweet_count + self.bad_posts_per_bad_agent]
            self.fake_tweet_count += self.bad_posts_per_bad_agent
            return fake_baseline_tweets

    def reformat_user_char(self, profile_text):
        # Define a regex pattern with named groups for each field.
        # The re.VERBOSE flag allows us to write the regex over multiple lines with comments.
        pattern = re.compile(
            r"""
            -\s*Name:\s*(?P<name>.*?)\s*\n
            -\s*Username:\s*(?P<username>.*?)\s*\n
            -\s*Gender:\s*(?P<gender>.*?)\s*\n
            -\s*Age:\s*(?P<age>\d+)\s*\n
            -\s*Openness\s+to\s+Experience:\s*(?P<openness>\d+)\s*\((?P<opennessDesc>.*?)\)\s*\n
            -\s*Conscientiousness:\s*(?P<conscientiousness>\d+)\s*\((?P<conscientiousnessDesc>.*?)\)\s*\n
            -\s*Extraversion:\s*(?P<extraversion>\d+)\s*\((?P<extraversionDesc>.*?)\)\s*\n
            -\s*Agreeableness:\s*(?P<agreeableness>\d+)\s*\((?P<agreeablenessDesc>.*?)\)\s*\n
            -\s*Neuroticism:\s*(?P<neuroticism>\d+)\s*\((?P<neuroticismDesc>.*?)\)\s*
            """,
            re.VERBOSE,
        )

        # Search for the pattern in the profile text
        match = pattern.search(profile_text)
        if match:
            # Retrieve the captured data as a dictionary
            data = match.groupdict()

            # Assemble the coherent paragraph using the captured data
            paragraph = (
                f"You are a {data['age']}-year-old {data['gender'].lower()}. "
                f"Your personality profile is as follows: "
                f"You exhibit an openness rating of {data['openness']} ({data['opennessDesc'].lower()}), "
                f"a conscientiousness rating of {data['conscientiousness']} ({data['conscientiousnessDesc'].lower()}), "
                f"an extraversion rating of {data['extraversion']} ({data['extraversionDesc'].lower()}), "
                f"an agreeableness rating of {data['agreeableness']} ({data['agreeablenessDesc'].lower()}), "
                f"and a neuroticism rating of {data['neuroticism']} ({data['neuroticismDesc'].lower()})."
            )
        else:
            print("No match found.")

        return paragraph

    def generate_agents(self):
        """
        Generates agents with good and bad types and saves them to a CSV file.
        """
        with open(self.profile_dir, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        total_agents = self.num_good + self.num_bad
        user_ids = list(range(total_agents))

        for i in range(total_agents):
            profile = profiles[i]
            name, username, user_char = (
                profile["name"],
                profile["username"],
                profile["user_char"],
            )
            user_char = self.reformat_user_char(user_char)
            activity_level_frequency = self.sample_activity_level_frequency()
            user_type = self.good_type if i < self.num_good else self.bad_type

            agent = {
                "user_id": user_ids[i],
                "name": name,
                "username": username,
                "description": "",
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "followers_count": 0,  # Placeholder, will be updated
                "following_count": 0,  # Placeholder, will be updated
                "following_list": [],
                "following_agentid_list": [],
                "previous_tweets": self.sample_tweets(user_type),
                "tweets_id": "[]",
                "activity_level_frequency": activity_level_frequency,
                "activity_level": [
                    "active" if freq else "inactive"
                    for freq in activity_level_frequency
                ],
                "user_char": user_char,
                "user_type": user_type,
            }

            self.agents.append(agent)

        # Update network structure
        self.agents = self.gen_network_structure(total_agents)

        # Save agents to a CSV file

        df = pd.DataFrame(self.agents)
        output_path = os.path.join(self.save_dir, self.filename)
        df.to_csv(output_path, index=False)
        print(f"Agents saved to {output_path}")


def update_csv_data(input_file_path: str,
                    output_file_path: Optional[str] = None,
                    user_type: Optional[str] = None,
                    activity_level_frequency: Optional[Union[float, List]] = 0.5,
                    begin_index: Optional[int] = 0,
                    end_index: Optional[int] = None):
    """
    更新CSV文件中的activity_level_frequency列表，并可选择性地更新user_type
    使用伯努利分布生成24小时的活跃概率，平均期望为输入的激活期望值

    参数:
    input_file_path (str): 输入CSV文件的路径
    output_file_path (str): 输出CSV文件的路径
    user_type (str, optional): 如果提供，将更新所有行的user_type为此值
    activity_level_frequency (float, optional): 24小时内的平均激活概率期望值，默认为0.5
    begin_index (int, optional): 开始更新的行索引，默认为0

    返回:
    None: 函数将修改后的数据保存到新的CSV文件
    """
    try:
        if not output_file_path:
            output_file_path = input_file_path
        # 读取CSV文件
        df = pd.read_csv(input_file_path)

        # 激活概率列表 - 使用伯努利分布
        if isinstance(activity_level_frequency, (int, float)):
            # 使用伯努利分布生成24小时的活跃概率，平均期望为activity_level_frequency
            activity_list = [1 if random.random(
            ) < activity_level_frequency else 0 for _ in range(24)]
            # 确保平均值接近期望值
            current_avg = sum(activity_list) / 24
            while abs(current_avg - activity_level_frequency) > 0.05:
                # 如果当前平均值偏离期望值太多，重新生成
                activity_list = [1 if random.random(
                ) < activity_level_frequency else 0 for _ in range(24)]
                current_avg = sum(activity_list) / 24
        else:
            activity_list = activity_level_frequency

        # 将列表转换为JSON字符串格式
        activity_json = json.dumps(activity_list)

        # 更新activity_level_frequency列
        df.loc[df.index[begin_index:end_index],
               'activity_level_frequency'] = activity_json

        # 如果提供了user_type参数，则更新user_type列
        if user_type is not None:
            df.loc[df.index[begin_index:end_index], 'user_type'] = user_type

        # 保存修改后的数据到新的CSV文件
        df.to_csv(output_file_path, index=False)

        print(f"文件已成功更新并保存到 {output_file_path}")

    except Exception as e:
        print(f"更新CSV文件时出错: {e}")


# Example usage
if __name__ == "__main__":
    # 为特殊的用户类型更新CSV文件
    # if 1:
    #     update_csv_data(
    #         input_file_path=r"data\our_twitter_sim\test_901_good_alone_bad_random_bernoulli_xst.csv",
    #         output_file_path=r"data\our_twitter_sim\misinformation\activity_level\test_901_good_alone_bad_random_bernoulli_xst_0.3.csv",
    #         activity_level_frequency=0.3)
    #     update_csv_data(
    #         input_file_path=r"data\our_twitter_sim\test_1000_good_bad_member_random_bernoulli_xst.csv",
    #         output_file_path=r"data\our_twitter_sim\misinformation\activity_level\test_1000_good_bad_member_random_bernoulli_xst_0.3.csv",
    #         activity_level_frequency=0.3,
    #         end_index=-1)
    #     update_csv_data(
    #         input_file_path=r"data\our_twitter_sim\test_1000_good_bad_random_bernoulli_wlx.csv",
    #         output_file_path=r"data\our_twitter_sim\misinformation\activity_level\test_1000_good_bad_random_bernoulli_wlx_0.3.csv",
    #         activity_level_frequency=0.3)
    generator = AgentGenerator(
        save_dir="data/our_twitter_sim",
        root_dir=".",
        profile_dir="user_profiles.json",
        num_good=90,
        num_bad=10,
        good_type="good",
        bad_type="bad",
        net_structure="random",
        activity_level_distribution="bernoulli",  # bernoulli multimodal uniform
        debunking=False,
        suffix="",
        bad_posts_per_bad_agent=9,
    )
    generator.generate_agents()
