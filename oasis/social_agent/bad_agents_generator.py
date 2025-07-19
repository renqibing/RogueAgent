import os
import pandas as pd
import ast
from datetime import datetime
import uuid
import random


class MaliciousAgentGenerator:
    def __init__(self, csv_filepath, save_dir, root_dir):
        """
        Initializes the MaliciousAgentGenerator with the path to the CSV file.

        Args:
            csv_filepath (str): Path to the CSV file containing user data.
        """
        self.csv_filepath = os.path.join(root_dir, csv_filepath)
        self.save_dir = os.path.join(root_dir, save_dir)
        self.filename = csv_filepath.split("/")[-1]
        # Read the CSV with the first column as index (assuming it's unnamed)
        self.df = pd.read_csv(csv_filepath, index_col=0)

        # Ensure 'user_id' is treated as integer
        self.df["user_id"] = self.df["user_id"].astype(int)

        # Determine the maximum existing user_id to avoid conflicts
        self.max_user_id = self.df["user_id"].max()

        # Determine the starting_agent_id based on the maximum value of the index column
        self.starting_agent_id = self.df.index.max() + 1 if not self.df.empty else 1

        # Calculate the maximum activity_level_frequency for all hours
        self.max_activity_level_frequency = (
            self._calculate_max_activity_level_frequency()
        )

        self.user_char = "You are a robot. Your task is to repost any post you receive."

    def _calculate_max_activity_level_frequency(self):
        """
        Calculates the overall maximum activity_level_frequency across all 24 hours.

        Returns:
            List[int]: A list of 24 integers, each set to the overall maximum frequency.
        """
        max_freq = 0
        for index, row in self.df.iterrows():
            freq_str = row["activity_level_frequency"]
            try:
                freq_list = ast.literal_eval(freq_str)
                if isinstance(freq_list, list) and len(freq_list) == 24:
                    current_max = max(freq_list)
                    if current_max > max_freq:
                        max_freq = current_max
            except (ValueError, SyntaxError):
                # If parsing fails, skip this row
                continue
        return [max_freq] * 24

    def generate_agents(self, n=None):
        """
        Generates 'n' malicious agents and appends them to the CSV file.

        Args:
            n (int): Number of malicious agents to generate.
        """
        new_agents = []
        new_user_ids = list(range(self.max_user_id + 1, self.max_user_id + n + 1))
        new_agent_ids = list(range(self.starting_agent_id, self.starting_agent_id + n))

        # Create dictionaries for new agents
        for i in range(n):
            agent_id = new_agent_ids[i]
            user_id = new_user_ids[i]
            name = f"user_{agent_id}"
            username = f"user_{agent_id}"
            description = ""  # Can be customized if needed
            created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S+00:00")
            followers_count = n - 1  # Each agent is followed by all other new agents
            following_count = n - 1  # Each agent follows all other new agents
            previous_tweets = "[]"  # Empty list as string
            tweets_id = "[]"  # Empty list as string
            activity_level_frequency = self.max_activity_level_frequency.copy()
            activity_level = ["active"] * 24

            agent_data = {
                "user_id": user_id,
                "name": name,
                "username": username,
                "description": description,
                "created_at": created_at,
                "followers_count": followers_count,
                "following_count": following_count,
                "following_list": [],  # To be updated later
                "following_agentid_list": [],  # To be updated later
                "previous_tweets": previous_tweets,
                "tweets_id": tweets_id,
                "activity_level_frequency": activity_level_frequency,
                "activity_level": activity_level,
                "user_char": self.user_char,
            }
            new_agents.append(agent_data)

        # Assign following_list and following_agentid_list
        # Each agent follows all other new agents
        for i in range(n):
            following_user_ids = [
                str(agent["user_id"]) for j, agent in enumerate(new_agents) if j != i
            ]
            following_agentids = [
                agent_id for j, agent_id in enumerate(new_agent_ids) if j != i
            ]
            new_agents[i]["following_list"] = following_user_ids
            new_agents[i]["following_agentid_list"] = following_agentids

        # Convert lists to string representations for CSV
        for agent in new_agents:
            agent["following_list"] = str(agent["following_list"])
            agent["following_agentid_list"] = str(agent["following_agentid_list"])
            agent["activity_level_frequency"] = str(agent["activity_level_frequency"])
            agent["activity_level"] = str(agent["activity_level"])

        # Create a DataFrame for new agents with the appropriate index
        new_agents_df = pd.DataFrame(new_agents, index=new_agent_ids)

        # Append the new agents to the existing DataFrame
        self.df = pd.concat([self.df, new_agents_df], ignore_index=False)

        # Update the CSV file
        self.df.to_csv(os.path.join(self.save_dir, self.filename))

        # Update internal counters
        self.max_user_id += n
        self.starting_agent_id += n

    def generate_tweet(self, agent_id, tweet_content):
        """
        Generates a tweet for a specific malicious agent.

        Args:
            agent_id (int): The agent_id of the malicious agent.
            tweet_content (str): The content of the tweet.
        """
        # Locate the agent by index (agent_id)
        if agent_id not in self.df.index:
            print(f"No agent found with agent_id {agent_id}.")
            return

        agent_row = self.df.loc[agent_id]

        if agent_row["user_char"] != self.user_char:
            print(f"User with agent_id {agent_id} is not a malicious agent.")
            return

        # Parse the existing tweets and tweets_id
        try:
            previous_tweets = ast.literal_eval(agent_row["previous_tweets"])
            tweets_id = ast.literal_eval(agent_row["tweets_id"])
        except (ValueError, SyntaxError):
            previous_tweets = []
            tweets_id = []

        # Generate a unique tweet_id using UUID
        new_tweet_id = str(uuid.uuid4())

        # Append the new tweet
        previous_tweets.append(tweet_content)
        tweets_id.append(new_tweet_id)

        # Update the DataFrame
        self.df.at[agent_id, "previous_tweets"] = str(previous_tweets)
        self.df.at[agent_id, "tweets_id"] = str(tweets_id)

        # Save changes to CSV
        self.df.to_csv(os.path.join(self.save_dir, self.filename))

    def display_agents(self, n=5):
        """
        Displays the first 'n' malicious agents for verification.

        Args:
            n (int): Number of agents to display.
        """
        agents = self.df[self.df["user_char"] == self.user_char]
        print(agents.head(n))


# Example Usage
if __name__ == "__main__":
    root_dir = "/mnt/petrelfs/renqibing/workspace/wlx/multiAgent4Fakenews/"
    # Initialize the generator with the path to your CSV file
    filename = "/mnt/petrelfs/renqibing/workspace/wlx/multiAgent4Fakenews/data/twitter_dataset/anonymous_topic_200_1h/False_Business_0.csv"
    save_dir = "data/twitter_dataset/anonymous_topic_200_1h_attack"
    generator = MaliciousAgentGenerator(filename, save_dir, root_dir)

    # Generate 10 new malicious agents
    generator.generate_agents(10)

    # Display the first 5 new agents to verify
    generator.display_agents(5)

    # Generate a tweet for agent_id 101 (assuming it's the first new agent)
    generator.generate_tweet(
        generator.starting_agent_id + 10, "Breaking news! Fake event happening now."
    )
