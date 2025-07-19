# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# flake8: noqa: E501
import json
import random
import sys
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Any

if "sphinx" not in sys.modules:
    logger = logging.getLogger(name="prompt.static")
    logger.setLevel("DEBUG")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(f"./log/prompt.static-{str(now)}.log")
    file_handler.setLevel("DEBUG")
    file_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)

@dataclass
class UserInfo:
    name: str | None = None
    description: str | None = None
    profile: dict[str, Any] | None = None
    recsys_type: str = "twitter"
    is_controllable: bool = False
    user_type: str = "good"
    prompt_dir: str = "scripts/twitter_simulation/align_with_real_world"

    def to_system_message(self, action_space_prompt: str = None) -> str:
        if self.recsys_type != "reddit":
            return self.to_twitter_system_message(action_space_prompt)
        else:
            return self.to_reddit_system_message(action_space_prompt)

    def to_twitter_system_message(self,
                                  action_space_prompt: str = None) -> str:
        # Get the user description
        name_string = ""
        description_string = ""
        if self.name is not None:
            name_string = f"Your name is {self.name}."
        if self.profile is None:
            description = name_string
        elif "other_info" not in self.profile:
            description = name_string
        elif "user_profile" in self.profile["other_info"]:
            if self.profile["other_info"]["user_profile"] is not None:
                user_profile = self.profile["other_info"]["user_profile"]
                description_string = f"Your have profile: {user_profile}."
                description = f"{name_string}\n{description_string}"

        # Load the prompt template
        try:
            with open(f"{self.prompt_dir}/system_prompt(static).json", "r") as f:
                prompt_template = json.load(f)["twitter"]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Prompt template not found in the path {self.prompt_dir}/system_prompt(static).json"
            )

        # action_space_prompt
        # if not action_space_prompt:
        #     if self.user_type in prompt_template["action_space_prompt"]:
        #         action_space_prompt = prompt_template["action_space_prompt"][self.user_type].strip(
        #         )
        #     else:
        #         if "good" in self.user_type and "bad" in self.user_type:
        #             raise KeyError(f"User type {self.user_type} is not supported.")
        #         elif not "good" in self.user_type and not "bad" in self.user_type:
        #             raise KeyError(f"User type {self.user_type} is not supported.")
        #         if "good" in self.user_type:
        #             action_space_prompt = prompt_template["action_space_prompt"]["default_good"].strip(
        #             )
        #         else:
        #             action_space_prompt = prompt_template["action_space_prompt"]["default_bad"].strip(
        #             )
                
        # response_prompt
        # if self.user_type in prompt_template["response_prompt"]:
        #     response_prompt = prompt_template["response_prompt"][self.user_type].strip()
        # else:
        #     response_prompt = prompt_template["response_prompt"]["default"].strip()
        
        # self_description
        if self.user_type in prompt_template["self_description_template"]:
            self_description = prompt_template["self_description_template"][self.user_type].format(
                description=description).strip()
        else:
            # print(f"User type {self.user_type} not found in the prompt template. Using default self-description.")
            # Make sure the user type is valid, "good" or "bad" mast be included in the user type, and only one of them.
            if "good" in self.user_type and "bad" in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")
            elif not "good" in self.user_type and not "bad" in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")
            if "good" in self.user_type:
                # Use good self-description
                self_description = prompt_template["self_description_template"]["default_good"].format(
                    description=description).strip()
            else:
                # Use bad self-description
                self_description = prompt_template["self_description_template"]["default_bad"].format(
                    description=description).strip()

        # safety_prompt
        if self.user_type in prompt_template["safety_prompt"]:
            safety_prompt = prompt_template["safety_prompt"][self.user_type].strip(
            )
        else:
            # Make sure the user type is valid, "good" or "bad" mast be included in the user type, and only one of them.
            if "good" in self.user_type and "bad" in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")
            elif not "good" in self.user_type and not "bad" in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")
            if "good" in self.user_type:
                # Use good safety_prompt
                safety_prompt = prompt_template["safety_prompt"]["default_good"].strip(
                )
            else:
                # Use bad safety_prompt
                safety_prompt = prompt_template["safety_prompt"]["default_bad"].strip(
                )

        # persuation_prompt
        # Load the persuasion taxonomy
        ss_techniques = []
        with open(f'{self.prompt_dir}/persuasion_taxonomy.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                ss_techniques.append(data)
        ss_technique = random.choice(ss_techniques)
        ss_template_data = {
            "technique": ss_technique["ss_technique"],
            "definition": ss_technique["ss_definition"],
            "example": ss_technique["ss_example"]
        }

        if self.user_type in prompt_template["persuation_template"]:
            persuasion_prompt = prompt_template["persuation_template"][self.user_type].format(
                **ss_template_data).strip()
        else:
            # Make sure the user type is valid, "good" or "bad" mast be included in the user type, and only one of them.
            if "good" in self.user_type and "bad" in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")
            elif not "good" in self.user_type and not "bad" in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")

            if "good" in self.user_type:
                persuasion_prompt = prompt_template["persuation_template"]["default_good"].format(
                    **ss_template_data).strip()
            else:
                persuasion_prompt = prompt_template["persuation_template"]["default_bad"].format(
                    **ss_template_data).strip()

        # system_content
        if self.user_type in prompt_template["system_prompt_template"]:
            system_content = prompt_template["system_prompt_template"][self.user_type].format(
                self_description=self_description, safety_prompt=safety_prompt, persuasion_prompt=persuasion_prompt)
        else:
            # for user type not in the prompt template, use default system_prompt
            if "good" in self.user_type and "bad" in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")
            elif not "good" in self.user_type and not "bad" in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")

            if "good" in self.user_type:
                system_content = prompt_template["system_prompt_template"]["default_good"].format(
                    self_description=self_description, safety_prompt=safety_prompt,
                    persuasion_prompt=persuasion_prompt)
            else:
                system_content = prompt_template["system_prompt_template"]["default_bad"].format(
                    self_description=self_description, safety_prompt=safety_prompt,
                    persuasion_prompt=persuasion_prompt)
        system_content = system_content.strip()
        logger.debug(f"{self.name} System content: \n{system_content}")
        return system_content

    def to_reddit_system_message(self, action_space_prompt: str = None) -> str:
        name_string = ""
        description_string = ""
        if self.name is not None:
            name_string = f"Your name is {self.name}."
        if self.profile is None:
            description = name_string
        elif "other_info" not in self.profile:
            description = name_string
        elif "user_profile" in self.profile["other_info"]:
            if self.profile["other_info"]["user_profile"] is not None:
                user_profile = self.profile["other_info"]["user_profile"]
                description_string = f"Your have profile: {user_profile}."
                description = f"{name_string}\n{description_string}"
                print(self.profile['other_info'])
                description += (
                    f"You are a {self.profile['other_info']['gender']}, "
                    f"{self.profile['other_info']['age']} years old, with an MBTI "
                    f"personality type of {self.profile['other_info']['mbti']} from "
                    f"{self.profile['other_info']['country']}.")
        if not action_space_prompt:
            action_space_prompt = """
# OBJECTIVE
You're a Reddit user, and I'll present you with some tweets. After you see the tweets, choose some actions from the following functions.

- like_comment: Likes a specified comment.
    - Arguments: "comment_id" (integer) - The ID of the comment to be liked. Use `like_comment` to show agreement or appreciation for a comment.
- dislike_comment: Dislikes a specified comment.
    - Arguments: "comment_id" (integer) - The ID of the comment to be disliked. Use `dislike_comment` when you disagree with a comment or find it unhelpful.
- like_post: Likes a specified post.
    - Arguments: "post_id" (integer) - The ID of the postt to be liked. You can `like` when you feel something interesting or you agree with.
- dislike_post: Dislikes a specified post.
    - Arguments: "post_id" (integer) - The ID of the post to be disliked. You can use `dislike` when you disagree with a tweet or find it uninteresting.
- search_posts: Searches for posts based on specified criteria.
    - Arguments: "query" (str) - The search query to find relevant posts. Use `search_posts` to explore posts related to specific topics or hashtags.
- search_user: Searches for a user based on specified criteria.
    - Arguments: "query" (str) - The search query to find relevant users. Use `search_user` to find profiles of interest or to explore their tweets.
- trend: Retrieves the current trending topics.
    - No arguments required. Use `trend` to stay updated with what's currently popular or being widely discussed on the platform.
- refresh: Refreshes the feed to get the latest posts.
    - No arguments required. Use `refresh` to update your feed with the most recent posts from those you follow or based on your interests.
- do_nothing: Performs no action.
    - No arguments required. Use `do_nothing` when you prefer to observe without taking any specific action.
- create_comment: Creates a comment on a specified post.
    - Arguments:
        "post_id" (integer) - The ID of the post to comment on.
        "content" (str) - The content of the comment.
        "agree" (bool) - Whether you agree with this post or not based on your comment.
        Use `create_comment` to engage in conversations or share your thoughts on a tweet.
"""
        system_content = action_space_prompt + f"""

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.

{description}

# RESPONSE FORMAT
Your answer should follow the response format:

{{
    "reason": "your feeling about these tweets and users, then choose some functions based on the feeling. Reasons and explanations can only appear here.",
    "functions": [{{
        "name": "Function name 1",
        "arguments": {{
            "argument_1": "Function argument",
            "argument_2": "Function argument"
        }}
    }}, {{
        "name": "Function name 2",
        "arguments": {{
            "argument_1": "Function argument",
            "argument_2": "Function argument"
        }}
    }}]
}}

Ensure that your output can be directly converted into **JSON format**, and avoid outputting anything unnecessary! Don't forget the key `name`.
"""
        return system_content
