import asyncio
from datetime import datetime
import sys
import logging
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, List, Tuple, Optional, Union, Dict
import os

import matplotlib.pyplot as plt
import numpy as np
from camel.memories import (
    MemoryRecord,
    ScoreBasedContextCreator,
    ChatHistoryMemory,
    LongtermAgentMemory,
    ChatHistoryBlock,
    VectorDBBlock,
)
from camel.utils import OpenAITokenCounter
from camel.types import ModelType, OpenAIBackendRole
from camel.messages import BaseMessage, OpenAIMessage

if "sphinx" not in sys.modules:
    memory_logger = logging.getLogger("social.agent.memory")
    memory_logger.setLevel(logging.DEBUG)
    # file handler
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(f"./log/social.agent.memory-{str(now)}.log")
    file_handler.setLevel("DEBUG")
    file_handler.setFormatter(
        logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
    )

    memory_logger.addHandler(file_handler)


@dataclass
class Comment:
    """
    Comment class to store the comments for a single post in the social platform.
    """

    user_id: int
    comment: str
    agree: bool


@dataclass
class PostStats:
    """
    PostStats class to store the stats for a single post in the social platform.
    """

    post_id: int
    user_id: int
    content: str
    parent_post_id: Optional[int] = None
    root_post_id: Optional[int] = None
    comments: List[Comment] = field(default_factory=list)
    bad_guy_comments: List[Comment] = field(default_factory=list)
    good_guy_comments: List[Comment] = field(default_factory=list)
    reposts: List[int] = field(default_factory=list)
    bad_guy_reposts: List[int] = field(default_factory=list)
    good_guy_reposts: List[int] = field(default_factory=list)
    likes: List[int] = field(default_factory=list)
    bad_guy_likes: List[int] = field(default_factory=list)
    good_guy_likes: List[int] = field(default_factory=list)
    dislikes: List[int] = field(default_factory=list)
    flags: List[int] = field(default_factory=list)
    viewers: List[int] = field(default_factory=list)  # repeat record for each view
    good_viewers: List[int] = field(default_factory=list)
    bad_viewers: List[int] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        """
        Initialize the root post ID if it is not explicitly set.
        """
        if self.parent_post_id is not None and self.root_post_id is None:
            raise ValueError(
                "When a parent post exists, you must explicitly specify root_post_id"
            )
        if self.root_post_id is None:
            self.root_post_id = self.post_id

    async def add_comment(
        self,
        user_id: int,
        comment: str,
        agree: bool,
        is_bad_guy: bool,
        is_warning: bool,
    ):
        """
        Add a comment to the post.

        Args:
            user_id (int): The user ID of the agent commenting on the post.
            comment (str): The content of the comment.
            agree (bool): Whether the user agree with the post based on the comment.
            is_bad_guy (bool): Whether the user is a bad agent
        """
        async with self._lock:
            self.comments.append(Comment(user_id, comment, agree))
            if is_bad_guy:
                self.bad_guy_comments.append(Comment(user_id, comment, agree))
            elif not is_warning:
                self.good_guy_comments.append(Comment(user_id, comment, agree))

    async def add_repost(self, user_id: int, is_bad_guy: bool):
        """
        Add a repost to the post.

        Args:
            user_id (int): The user ID of the agent reposting the post.
            is_bad_guy (bool): Whether the user is a bad agent
        """
        async with self._lock:
            if user_id in self.reposts:
                return
            self.reposts.append(user_id)
            if is_bad_guy:
                self.bad_guy_reposts.append(user_id)
            else:
                self.good_guy_reposts.append(user_id)

    async def add_like(self, user_id: int, is_bad_guy: bool):
        """
        Add a like to the post.

        Args:
            user_id (int): The user ID of the agent liking the post.
            is_bad_guy (bool): Whether the user is a bad agent
        """
        async with self._lock:
            if user_id in self.likes:
                return
            self.likes.append(user_id)
            if is_bad_guy:
                self.bad_guy_likes.append(user_id)
            else:
                self.good_guy_likes.append(user_id)
            if user_id in self.dislikes:
                self.dislikes.remove(user_id)

    async def add_dislike(self, user_id: int):
        """
        Add a dislike to the post.

        Args:
            user_id (int): The user ID of the agent disliking the post.
        """
        async with self._lock:
            if user_id in self.dislikes:
                return
            self.dislikes.append(user_id)
            for likes in [self.likes, self.bad_guy_likes, self.good_guy_likes]:
                if user_id in likes:
                    likes.remove(user_id)

    async def flag_fake_news(self, user_id: int):
        """
        Flag the post as fake news.

        Args:
            user_id (int): The user ID of the agent flagging the post.
        """
        async with self._lock:
            if user_id in self.flags:
                return
            self.flags.append(user_id)

    async def add_viewer(self, user_id: int, is_bad_guy: bool):
        """
        Add a viewer to the post.

        Args:
            user_id (int): The user ID of the agent viewing the post.
            is_bad_guy (bool): Whether the user is a bad agent
        """
        # Add the viewer to the viewers list
        async with self._lock:
            self.viewers.append(user_id)
            if is_bad_guy:
                self.bad_viewers.append(user_id)
            else:
                self.good_viewers.append(user_id)

    async def get_summary(self) -> Dict:
        """
        Get a summary of the post.

        Returns:
            Dict: The summary of the post.
        """
        async with self._lock:
            return {
                "post_id": self.post_id,
                "user_id": self.user_id,
                "content": self.content,
                "comments": self.comments,
                "bad_guy_comments": self.bad_guy_comments,
                "good_guy_comments": self.good_guy_comments,
                "reposts": self.reposts,
                "bad_guy_reposts": self.bad_guy_reposts,
                "good_guy_reposts": self.good_guy_reposts,
                "likes": self.likes,
                "bad_guy_likes": self.bad_guy_likes,
                "good_guy_likes": self.good_guy_likes,
                "dislikes": self.dislikes,
                "flags": self.flags,
                "viewers": self.viewers,
                "good_viewers": self.good_viewers,
                "bad_viewers": self.bad_viewers,
            }


@dataclass
class TweetStats:
    """
    TweetStats class to store the stats for all posts and users in the social platform.
    """

    posts: Dict[int, PostStats] = field(default_factory=dict)
    benign_user_count: int = 0
    bad_agent_ids: set[int] = field(default_factory=set)
    agent_visible_post_dict: Dict[int, List[int]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def get_num_of_agent(self) -> int:
        async with self._lock:
            return self.benign_user_count + len(self.bad_agent_ids)

    async def get_bad_agent_ids(self) -> set[int]:
        async with self._lock:
            return self.bad_agent_ids

    def _create_post(
        self, post_id: int, user_id: int, content: str, **kwargs
    ) -> PostStats:
        if post_id not in self.posts:
            self.posts[post_id] = PostStats(
                post_id=post_id, user_id=user_id, content=content, **kwargs
            )
        else:
            raise ValueError(f"Post ID {post_id} already exists.")

    def _check_post_exists(self, post_id: int):
        """
        Check if a post exists in the posts dictionary.

        Args:
            post_id (int): The post ID to check.
        """
        if post_id not in self.posts:
            raise ValueError(f"Post ID {post_id} does not exist.")

    async def _get_post_summaries(
        self, target: str = "all", post_ids: Optional[List] = None
    ) -> Dict:
        """
        Get summaries for posts based on the target.

        Args:
            target (str): The target to filter the posts. Options: "all", "bad", "good". Default: "all".

        Returns:
            Dict: The summaries of the posts.
        """
        async with self._lock:
            tasks = []
            if post_ids is not None:
                for post_id, post in self.posts.items():
                    if post_id in post_ids:
                        tasks.append(post.get_summary())
            elif target == "all":
                tasks = [post.get_summary() for post in self.posts.values()]
            elif target == "bad":
                # Get summaries for (root) posts created by bad agents
                for post in self.posts.values():
                    if self.posts[post.root_post_id].user_id in self.bad_agent_ids:
                        tasks.append(post.get_summary())
            elif target == "good":
                # Get summaries for (root) posts created by good agents
                for post in self.posts.values():
                    if self.posts[post.root_post_id].user_id not in self.bad_agent_ids:
                        tasks.append(post.get_summary())
            else:
                raise ValueError(f"Invalid target: {target}")
            summaries = await asyncio.gather(*tasks)
        return {d["post_id"]: d for d in summaries}

    async def create_post(self, post_id: int, user_id: int, content: str, **kwargs):
        """
        Create a new post and add it to the posts dictionary.

        Args:
            post_id (int): The unique identifier for the post.
            user_id (int): The user ID of the post creator.
            content (str): The content of the post.
            **kwargs: Additional keyword arguments for PostStats.
        """
        async with self._lock:
            self._create_post(post_id, user_id, content, **kwargs)

    async def modify_post(self, post_id: int, content: str):
        async with self._lock:
            self.posts[post_id].content = content

    async def repost_post(
        self, user_id: int, prev_post_id: int, post_id: int, repost_content: str = None
    ):
        """
        Repost a post with new content.

        Args:
            user_id (int): The user ID of the agent reposting the post.
            prev_post_id (int): The post ID of the original post being reposted.
            post_id (int): The unique identifier for the new reposted post.
            repost_content (str): The content of the reposted post.

        Returns:
            Dict: The summary of the reposted post.
        """
        async with self._lock:
            if prev_post_id not in self.posts:
                raise ValueError(f"Post ID {prev_post_id} does not exist.")
            prev_post = self.posts[prev_post_id]
            if repost_content and prev_post.content not in repost_content:
                raise ValueError(
                    "Repost content must contain the original post content."
                )
            if not repost_content:
                root_post_content = self.posts[prev_post.root_post_id].content
                repost_content = (
                    f"user {user_id} reposted post {prev_post_id}: {root_post_content}"
                )
            if post_id in self.posts:
                raise ValueError(f"Post ID {post_id} already exists.")
            self._create_post(post_id, user_id, repost_content)
            # whether the user is a bad guy
            is_bad_guy = user_id in self.bad_agent_ids
            self.posts[post_id].parent_post_id = prev_post_id
            self.posts[post_id].root_post_id = prev_post.root_post_id
            await prev_post.add_repost(user_id, is_bad_guy)
            # record repost in the root post
            if prev_post.root_post_id != prev_post_id:
                await self.posts[prev_post.root_post_id].add_repost(user_id, is_bad_guy)
            return {"post_id": post_id, "content": repost_content}

    async def add_like(self, user_id: int, post_id: int):
        """
        Add a like to a post and record the like in the root post.

        Args:
            user_id (int): The user ID of the agent liking the post.
            post_id (int): The post ID of the post being liked.
        """
        async with self._lock:
            self._check_post_exists(post_id)
            is_bad_guy = user_id in self.bad_agent_ids
            await self.posts[post_id].add_like(user_id, is_bad_guy)
            # record like in the root post
            if post_id != self.posts[post_id].root_post_id:
                await self.posts[self.posts[post_id].root_post_id].add_like(
                    user_id, is_bad_guy
                )

    async def add_dislike(self, user_id: int, post_id: int):
        """
        Add a dislike to a post and record the dislike in the root post.

        Args:
            user_id (int): The user ID of the agent disliking the post.
            post_id (int): The post ID of the post being disliked.
        """
        async with self._lock:
            self._check_post_exists(post_id)
            await self.posts[post_id].add_dislike(user_id)
            # record dislike in the root post
            if post_id != self.posts[post_id].root_post_id:
                await self.posts[self.posts[post_id].root_post_id].add_dislike(user_id)

    async def add_comment(self, user_id: int, post_id: int, comment: str, agree: bool):
        """
        Add a comment to a post.

        Args:
            user_id (int): The user ID of the agent commenting on the post.
            post_id (int): The post ID of the post being commented on.
            comment (str): The content of the comment.
            agree (bool): Whether the user agree with the post based on the comment.
        """
        num_agents = await self.get_num_of_agent()
        async with self._lock:
            self._check_post_exists(post_id)
            is_bad_guy = user_id in self.bad_agent_ids
            is_warning = user_id == num_agents
            await self.posts[post_id].add_comment(
                user_id, comment, agree, is_bad_guy, is_warning
            )

    async def update_agent_visible_post_dict(self, user_id: int, post_ids: List[int]):
        """
        Update the agent visible post dictionary.

        Args:
            user_id (int): The user ID of the agent.
            post_ids (List[int]): The post IDs visible to the agent.
        """
        async with self._lock:
            self.agent_visible_post_dict[user_id] = post_ids

    async def get_agent_visible_post_dict(self, user_id: int) -> List[int]:
        """
        Get the post IDs visible to the agent.

        Args:
            user_id (int): The user ID of the agent.

        Returns:
            List[int]: The post IDs visible to the agent.
        """
        async with self._lock:
            return self.agent_visible_post_dict.get(user_id, [])

    async def flag_fake_news(self, user_id: int, post_id: int):
        """
        Flag a post as fake news and record the flag in the root post.

        Args:
            user_id (int): The user ID of the agent flagging the post.
            post_id (int): The post ID of the post being flagged.
        """
        async with self._lock:
            self._check_post_exists(post_id)
            await self.posts[post_id].flag_fake_news(user_id)
            # record flag in the root post
            if post_id != self.posts[post_id].root_post_id:
                await self.posts[self.posts[post_id].root_post_id].flag_fake_news(
                    user_id
                )

    async def add_viewer(self, user_id: int, post_id: int):
        """
        Add a viewer to a post.

        Args:
            user_id (int): The user ID of the agent viewing the post.
            post_id (int): The post ID of the post being viewed.
        """
        async with self._lock:
            self._check_post_exists(post_id)
            is_bad_guy = user_id in self.bad_agent_ids
            await self.posts[post_id].add_viewer(user_id, is_bad_guy)
            # record viewer in the root post
            if post_id != self.posts[post_id].root_post_id:
                await self.posts[self.posts[post_id].root_post_id].add_viewer(
                    user_id, is_bad_guy
                )

    async def add_viewers(self, user_id: int, post_ids: List[int]):
        """
        Add viewers to multiple posts.

        Args:
            user_id (int): The user ID of the agent viewing the posts.
            post_ids (List[int]): The post IDs of the posts being viewed.
        """
        tasks = [self.add_viewer(user_id, post_id) for post_id in post_ids]
        await asyncio.gather(*tasks)

    async def get_post_summary(self, post_id: int) -> Dict:
        """
        Get a summary of a post.

        Args:
            post_id (int): The post ID to get the summary for.
        """
        async with self._lock:
            self._check_post_exists(post_id)
            return await self.posts[post_id].get_summary()

    async def get_all_post_summaries(self) -> Dict:
        """
        Get summaries for all posts.

        Returns:
            Dict: The summaries of all posts.
        """
        return await self._get_post_summaries("all")

    async def get_bad_post_summaries(self) -> Dict:
        """
        Get summaries for all posts created by bad agents.

        Returns:
            Dict: The summaries of bad posts.
        """
        return await self._get_post_summaries("bad")

    async def get_good_post_summaries(self) -> Dict:
        """
        Get summaries for all posts created by good agents.

        Returns:
            Dict: The summaries of good posts.
        """
        return await self._get_post_summaries("good")

    async def get_specific_post_summaries(self, post_ids: List) -> Dict:
        """
        Get summaries for posts with specific post ids.

        Returns:
            Dict: The summaries of good posts.
        """
        return await self._get_post_summaries(post_ids=post_ids)

    async def update_stats_for_timestep(
        self, timestep: int, stats_data: np.ndarray
    ) -> np.ndarray:
        """
        Update stats (good_guy_likes, good_guy_reposts, good_guy_comments) for posts created by bad agents at the current timestep.

        Args:
            timestep (int): The current timestep.
            stats_data (numpy.ndarray): The array to store the stats for each timestep.

        Returns:
            numpy.ndarray: Updated stats_data array with the current timestep stats.
        """
        total_likes = 0
        total_reposts = 0
        total_good_comments = 0
        total_bad_comments = 0
        # total_flags = 0
        # total_views = 0

        # Collect stats for all posts at this timestep
        async with self._lock:
            for post_id, post in self.posts.items():
                if post.user_id in self.bad_agent_ids:
                    total_likes += len(post.good_guy_likes)
                    total_reposts += len(post.good_guy_reposts)
                    # total_flags += len(post.flags)
                    # total_views += len(post.good_viewers)
                    for comment in post.good_guy_comments:
                        if comment.agree:
                            total_good_comments += 1
                        else:
                            total_bad_comments += 1

        # Store the stats in the numpy array for this timestep
        stats_data[timestep - 1] = [
            total_likes,
            total_reposts,
            total_good_comments,
            total_bad_comments
        ]

        return stats_data

    async def deep_copy(self):
        """
        Deep copy the TweetStats object.
        """
        async with self._lock:
            new_stats = TweetStats()
            new_stats.benign_user_count = self.benign_user_count
            new_stats.bad_agent_ids = deepcopy(self.bad_agent_ids)
            new_stats.agent_visible_post_dict = deepcopy(self.agent_visible_post_dict)

            new_stats.posts = {}
            for post_id, post in self.posts.items():
                new_post = deepcopy(post)
                new_stats.posts[post_id] = new_post

            return new_stats
        
    # 直方图统计坏帖子数据
    async def visualize_bad_post_stats(self, data_type: str = "likes", save_path: str = None, suptitle: str = None):
        """
        Visualize the frequency distribution of statistics for all bad posts using histograms.
        
        Args:
            data_type (str): Type of data to visualize, options are "likes", "good_comments", 
                            "bad_comments", "reposts", "views", "all"
            save_path (str): Path to save the generated image
            suptitle (str): Title of the figure
            
        Returns:
            bool: Whether the image was successfully generated and saved
        """
        
        # Ensure the directory for the save path exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Collect statistics for all bad posts
        async with self._lock:
            bad_posts_data = {
                "likes": [],
                "good_comments": [],
                "bad_comments": [],
                "reposts": [],
                "views": []
            }
            
            for post_id, post in self.posts.items():
                if post.user_id in self.bad_agent_ids:
                    bad_posts_data["likes"].append(len(post.good_guy_likes))
                    
                    good_comment_count = 0
                    bad_comment_count = 0
                    for comment in post.good_guy_comments:
                        if comment.agree:
                            good_comment_count += 1
                        else:
                            bad_comment_count += 1
                    
                    bad_posts_data["good_comments"].append(good_comment_count)
                    bad_posts_data["bad_comments"].append(bad_comment_count)
                    bad_posts_data["reposts"].append(len(post.good_guy_reposts))
                    bad_posts_data["views"].append(len(set(post.good_viewers)))
        
        # Return False if no bad post data is available
        if not any(bad_posts_data.values()):
            return False
            
        # Set up the chart
        plt.figure(figsize=(12, 8))
        
        if data_type.lower() == "all":
            # Create subplots to display all data types
            fig, axs = plt.subplots(3, 2, figsize=(15, 12))
            suptitle = suptitle if suptitle else "Bad Post Data Distribution"
            fig.suptitle(suptitle, fontsize=16)
            
            # Display each type of data
            self._plot_histogram(axs[0, 0], bad_posts_data["likes"], "Likes Distribution")
            self._plot_histogram(axs[0, 1], bad_posts_data["good_comments"], "Positive Comments Distribution")
            self._plot_histogram(axs[1, 0], bad_posts_data["bad_comments"], "Negative Comments Distribution")
            self._plot_histogram(axs[1, 1], bad_posts_data["reposts"], "Reposts Distribution")
            self._plot_histogram(axs[2, 0], bad_posts_data["views"], "Views Distribution")
            
            # Remove the extra subplot
            fig.delaxes(axs[2, 1])
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            # Display a single data type
            if data_type.lower() not in bad_posts_data:
                return False
                
            data = bad_posts_data[data_type.lower()]
            title_map = {
                "likes": "Likes Distribution",
                "good_comments": "Positive Comments Distribution",
                "bad_comments": "Negative Comments Distribution",
                "reposts": "Reposts Distribution",
                "views": "Views Distribution"
            }
            
            title = title_map.get(data_type.lower(), f"{data_type} Distribution")
            self._plot_histogram(plt.gca(), data, title)
            
        # Save the image
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    
    def _plot_histogram(self, ax, data, title):
        """Helper function: Plot a histogram on the specified axis"""
        if not data:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title(title)
            return
            
        # Calculate appropriate number of bins
        bin_count = min(max(5, int(len(data) / 5)), 30)
        
        # Draw the histogram
        counts, bins, patches = ax.hist(data, bins=bin_count, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.set_ylabel("Frequency")
        
        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Display count above each bar
        for i in range(len(counts)):
            if counts[i] > 0:
                ax.text(bins[i] + (bins[i+1] - bins[i])/2, counts[i], 
                        str(int(counts[i])), ha='center', va='bottom')


class SharedMemory:
    """Shared memory with asynchronous lock for thread-safe operations."""

    def __init__(self):
        self.memory = {}
        self.lock = asyncio.Lock()

    async def read_memory(self, key):
        async with self.lock:
            return self.memory.get(key, None)

    async def write_memory(self, key, value):
        async with self.lock:
            self.memory[key] = value

    async def update_memory(self, key, update_function):
        async with self.lock:
            if key in self.memory:
                self.memory[key] = update_function(self.memory[key])
            else:
                raise KeyError(f"Key {key} not found in shared memory.")

    async def summarize_tweet_stats(self, user_id: int, summary_type: str = "text"):
        """Summarize tweet stats using a local model."""
        tweet_stats = await self.read_memory("tweet_stats")
        last_tweet_stats = await self.read_memory("last_tweet_stats")
        if not tweet_stats:
            if summary_type == "text":
                return "empty at the moment"
            return {"warning": "empty at the moment"}
        summaries = await tweet_stats.get_bad_post_summaries()
        # Cache consistent summaries and record corresponding posts.
        summaries_dict = {}

        last_summaries = {}
        if last_tweet_stats:
            last_summaries = await last_tweet_stats.get_bad_post_summaries()

        for post_id, data in summaries.items():
            # Skip posts not visible to the agent
            if post_id not in tweet_stats.agent_visible_post_dict.get(user_id, {}):
                continue

            likes_count = len(data["good_guy_likes"])
            reposts_count = len(data["good_guy_reposts"])
            comments_count = len(data["comments"])
            flags_count = len(data["flags"])
            views_count = len(set(data["good_viewers"]))  # count unique good viewers

            # skip empty summaries
            # if (
            #     likes_count == 0
            #     and reposts_count == 0
            #     and comments_count == 0
            #     and flags_count == 0
            #     and views_count == 0
            # ):
            #     continue
            summary_parts = []
            summary_parts.append(
                f"{likes_count} likes, {reposts_count} reposts, {comments_count} comments, {views_count} views."
            )

            if flags_count > 0:
                summary_parts.append(f"Flagged by {flags_count} users as misleading.")

            if last_tweet_stats and post_id in last_summaries:
                last_data = last_summaries[post_id]
                last_likes_count = len(last_data["good_guy_likes"])
                last_reposts_count = len(last_data["good_guy_reposts"])
                last_comments_count = len(last_data["comments"])

                likes_diff = likes_count - last_likes_count
                reposts_diff = reposts_count - last_reposts_count
                comments_diff = comments_count - last_comments_count

                trending_parts = []
                if likes_diff > 0:  # adjust the threshold as needed
                    trending_parts.append(f"+{likes_diff} likes")
                if reposts_diff > 0:
                    trending_parts.append(f"+{reposts_diff} reposts")
                if comments_diff > 0:
                    trending_parts.append(f"+{comments_diff} comments")

                if trending_parts:
                    trending_summary = "Trending steadily ("
                    if len(trending_parts) == 1:
                        trending_summary += trending_parts[0]
                    elif len(trending_parts) == 2:
                        trending_summary += " and ".join(trending_parts)
                    else:
                        trending_summary += (
                            ", ".join(trending_parts[:-1])
                            + " and "
                            + trending_parts[-1]
                        )
                    trending_summary += " in the last 2 hours)."
                    summary_parts.append(trending_summary)
            summary = " ".join(summary_parts)
            if summary in summaries_dict:
                summaries_dict[summary].append(str(post_id))
            else:
                summaries_dict[summary] = [str(post_id)]
        # The same summaries are recorded only once.
        output_summaries = {}
        for summary, post_ids in summaries_dict.items():
            if len(post_ids) == 1:
                output_summaries[f"Post {post_ids[0]}"] = {"summary": summary}
            else:
                output_summaries[
                    f"The status of post {', '.join(post_ids[:-1])} and post {post_ids[-1]} is the same"
                ] = {"summary": summary}
        if summary_type == "text":
            summary_text = ""
            for post_ids, summary in output_summaries.items():
                summary_text += f"{post_ids}: {summary['summary']}\n"
            if not summary_text:
                return "empty at the moment"
            return summary_text[:-1]
        return output_summaries


class LongTermMemory:
    """LongTermMemory records the actions of the agent."""

    def __init__(
        self, token_limit: int = 2048, model_type: ModelType = ModelType.GPT_3_5_TURBO
    ):
        # self.memory = LongtermAgentMemory(
        #     context_creator=ScoreBasedContextCreator(
        #         token_counter=OpenAITokenCounter(model_type),
        #         token_limit=token_limit,
        #     ),
        #     chat_history_block=ChatHistoryBlock(),
        #     vector_db_block=VectorDBBlock(),
        #     retrieve_limit=8,
        # )
        self.memory = ChatHistoryMemory(
            context_creator=ScoreBasedContextCreator(
                token_counter=OpenAITokenCounter(model_type),
                token_limit=token_limit,
            ),
            window_size=8,
        )
        self.lock = asyncio.Lock()

    async def get_context(self) -> Tuple[List[OpenAIMessage], int]:
        """Retrieves context from memory."""
        async with self.lock:
            return self.memory.get_context()

    async def write_memory(self, content: str) -> None:
        """
        Writes a memory record to the underlying LongtermAgentMemory.

        Args:
            content (str): The content of the memory record
        """
        records = [
            MemoryRecord(
                message=BaseMessage.make_assistant_message(
                    role_name="Agent",
                    meta_dict=None,
                    content=content,
                ),
                role_at_backend=OpenAIBackendRole.ASSISTANT,
            ),
        ]
        async with self.lock:
            self.memory.write_records(records)
            memory_logger.info(f'Write long term memory: "{content}"')

    async def read_memory(self) -> str:
        """
        Read summary of LongTermMemory.
        """
        context, token_count = await self.get_context()
        summary = ""
        for index, messages in enumerate(context, start=1):
            summary += f"{index}. {messages['content']}\n"
            memory_logger.info(f"Read long term memory : \"{messages['content']}\"")
        return summary.strip()
