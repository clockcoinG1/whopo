import json
import os

import numpy as np
import openai
import pandas as pd
import requests
import tiktoken
import tqdm
from openai.embeddings_utils import cosine_similarity, get_embedding

from constants import (
    EMBEDDING_ENCODING,
    GPT_MODEL,
    TOKEN_MAX_SUMMARY,
    base,
    chat_base,
    headers,
    oai_api_key_embedder,
    proj_dir,
)
from utils import setup_logger

openai.api_key = oai_api_key_embedder
tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)
encoder = tokenizer


def generate_summary(
    df: pd.DataFrame,
    model: str = "text-davinci-003",
    proj_dir: str = proj_dir,
) -> pd.DataFrame:
    """
    Generate a summary of each file in the dataframe using the OpenAI API.

    Args:
                    df (pd.DataFrame): The dataframe containing the file information.
                    model (str): The name of the OpenAI API model to use.
                    proj_dir (str): The name of the project directory.

    Returns:
                    pd.DataFrame: The dataframe with the summaries added.
    """
    # df["file_path"] = df["file_path"].str.replace(
    # 		root_dir , ""
    # )
    logger = setup_logger("Summary Logger")
    message = ""
    try:
        if not model:
            model = "text-davinci-003"
    except NameError:
        model = "text-davinci-003"
        # model="code-davinci-002"
    try:
        if not model:
            model = "text-davinci-003"
    except NameError:
        model = "text-davinci-003"
    comp_type = "finish_reason" if not model or model != "codex" else "finish_details"
    for _, row in tqdm.tqdm(df.iterrows()):
        code = row["code"]
        filepath = row["file_path"]
        filename = row["file_name"]
        prompt = (
            "\nSYSTEM: You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are"
            " intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is"
            f" instructed. You always answer truthfully and don't make things up.\nUSER:{code}\nUSER:Please summarize"
            " the key features of the specified file within the project directory, and present the information in a"
            " concise bullet-point format. Since the code is chunked and you are using a moving window approach, focus"
            "on the codes snippet content and do not generalize since that will cause repetition. Focus on aspects such"
            "as the file's content.\nASSISTANT: Sure, here are the"
            f" key features of the `{filepath}` code snippet```\n -"
        )
        encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
        enc_prompt = encoder.encode(str(prompt))
        tokens = len(encoder.encode(code) + enc_prompt) or 1
        abs(7800 - tokens)
        if tokens >= 20:
            logger.info(f"Token length is {tokens} for {filepath} chunk")
            pass
        r = requests.post(
            base,
            headers=headers,
            stream=True,
            json={
                "model": model,
                "prompt": prompt,
                "temperature": 2,
                "top_p": 0.1,
                "stream": True,
                "n": 1,
                # "logit_bias": {
                # 	"[27, 91, 320, 62, 437, 91, 29, 198]" : -100
                # 	},
                # "stop": ["<|endoftext|>" , "\n\n\n"],
                "stop": ["\nSYSTEM:", "\nUSER:", "\nASSISTANT:", "<|im_end|>"],
                "max_tokens": int(TOKEN_MAX_SUMMARY) + tokens,
                "presence_penalty": 1,
                "frequency_penalty": 1,
            },
        )
        summary = ""
        print(f"\n\n\x1b[33m{filepath}\x1b[0m", end="\n", flush=True)
        for line in r.iter_lines():
            data = line.decode("utf-8")
            if data.startswith("data: ") and data != "data: [DONE]":
                data = json.loads(data[5:])
                if data["object"] == "text_completion":
                    if data["choices"][0][comp_type] == "stop":
                        break
                    else:
                        if data["choices"][0]["text"]:
                            print(data["choices"][0]["text"], flush=False, end="")
                            message += data["choices"][0]["text"]
                            summary += data["choices"][0]["text"]
                        else:
                            message += "\n"
                            print("\n", flush=True, end="\n")
                            message.strip()
                            continue
            try:
                df[df["file_name"] == filename]["summary"].values[0]
                df.loc[df["file_name"] == filename, "summary"] = f"{summary.strip()}"
            except KeyError:
                df.loc[df["file_name"] == filename, "summary"] = summary.strip()
    return df


def get_tokens(df, colname):
    EMBEDDING_ENCODING = "cl100k_base"
    encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
    list(df[df.columns[df.columns.to_series().str.contains(colname)]])
    df = ce.df
    for _, row in tqdm.tqdm(ce.df.iterrows()):
        print(row)

        filepath = row["file_path"]
        emb_data = "file path: " + filepath + "\n" + str(row[colname])
        tokens = len(encoder.encode(emb_data))
        df.loc[df['file_path'] == filepath, 'tokens_summary'] = tokens
    df[['tokens_summary']] = df[['tokens_summary']].applymap(np.int64)
    return df


def df_search(df, summary_query, n=3, pprint=True):
    embedding = get_embedding(engine="text-embedding-ada-002", text=summary_query)

    df = df.loc[df.summary_embedding.notnull(), 'summary_embedding']

    df['summary_similarities'] = df.summary_embedding.apply(lambda x: cosine_similarity(x, embedding))
    df['code_similarities'] = df.code.apply(lambda x: cosine_similarity(x, embedding))
    res2 = df.sort_values('code_similarities', ascending=False).head(n)
    res = df.sort_values('summary_similarities', ascending=False).head(n)
    res_str = ""
    for index, r in res.iterrows():
        res_str += f"Filename:\n{r[1].file_name}\nSummary:\n{r[1].summary}\n"
    for index, r in res2.iterrows():
        res_str += f"File:\b{r[1].file_name}\nCode:\n{r[1].code}\n"
    return res


def q_and_a(df, question="What isthe most important file", total=10, MAX_SECTION_LEN=7000) -> str:
    SEPARATOR = "<|im_sep|>"
    encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
    separator_len = len(encoder.encode(SEPARATOR))
    relevant_notes = df_search(df, question, total, pprint=True)
    relevant_notes = relevant_notes.sort_values('summary_similarities', ascending=False).head(total)
    chosen_sections = []
    chosen_sections_len = 0
    for _, row in relevant_notes.iterrows():
        notes_str = f"Path: {row['file_name']}\nSummary:\n{row['summary']}\nCode:\n{row['code']}\n"
        notes_str_len = len(encoder.encode(notes_str))
        if chosen_sections_len + separator_len + notes_str_len > MAX_SECTION_LEN:
            break
        chosen_sections.append(SEPARATOR + notes_str)
        chosen_sections_len += separator_len + notes_str_len

    chosen_sections_str = "".join(chosen_sections)
    print(f"Selected {len(chosen_sections)} document sections:")
    return f'''<|start_context|>\n Project notes to help assistant with answering query "{question}" \n context: {chosen_sections_str}\n<|end_context|>\n<|im_sep|>'''


def chatbot(df, prompt="What does this code do?", n=4):
    avail_tokens = len(encoder.encode(prompt))
    r = requests.post(
        chat_base,
        headers=headers,
        stream=True,
        json={
            "model": GPT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are"
                        " intelligent, helpful, and an expert developer, who always gives the correct answer and only"
                        " does what is instructed. You always answer truthfully and don't make things up."
                    ),
                },
                {"role": "user", "content": f"{prompt}"},
            ],
            "temperature": 2,
            "top_p": 0.05,
            "n": 1,
            "stop": ["\nSYSTEM:", "\nUSER:", "\nASSISTANT:", "<|im_end|>"],
            "stream": True,
            "max_tokens": 8000 - int(avail_tokens),
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )

    for line in r.iter_lines():
        message = ""
        if line:
            data = line
            if data == "[DONE]":
                break
            else:
                data = json.loads(data[5:])
                if data["object"] == "chat.completion.chunk":
                    if data["choices"][0]["finish_reason"] == "stop":
                        break
                    else:
                        if "content" in data["choices"][0]["delta"]:
                            message += data["choices"][0]["delta"]["content"]
                            print(data["choices"][0]["delta"]["content"], flush=True, end="")
                        else:
                            message += "\n"
    return message.strip()


def df_search_sum(df, summary_query, n=10, pprint=True, n_lines=20):
    """Search for a code snippet using a summary query. Returns the top N results, and optionally prints them to stdout.

    Args:
        df (DataFrame): DataFrame with code snippets
        summary_query (str): Summary query to use for search
        n (int, optional): Number of results to return. Defaults to 10.
        pprint (bool, optional): Whether to print the results to stdout. Defaults to True.
        n_lines (int, optional): Number of lines to print for each result. Defaults to 20.

    Returns:
        str: String with the top N results
    """
    logger = setup_logger()
    try:
        logger.info("Getting embeddings")
        embedding = get_embedding(engine="text-embedding-ada-002", text=summary_query)
        logger.info("Calculating summary similarities")
        df['summary_similarities'] = df.summary_embedding.apply(
            lambda x: cosine_similarity(x, embedding) if x is not None else 0.00
        )
        logger.info("Calculating code similarities")
        df['code_similarities'] = df.code_embedding.apply(
            lambda x: cosine_similarity(x, embedding) if x is not None else 0.00
        )
        logger.log(1, "Sorting results")
        indexes = abs(n // 2)
        res = df.sort_values('summary_similarities', ascending=False).head()
        res = pd.concat([res, df.sort_values('code_similarities', ascending=False).head(indexes)], ignore_index=True)

        res_str = ""
        if pprint:
            for r in res.iterrows():
                summary = "\n".join(r[1].summary.split("\n")[:n_lines])
                code = "\n".join(r[1].code.split("\n")[:n_lines])
                logger.info(f"File:{r[1].file_name}\nCode:\n{code}\nSummary:\n{summary}\n")
                res_str += summary + "\n" + code + "\n\n"

        return res_str
    except Exception as e:
        logger.error(f"Error found: {e}")
        return f"Error found: {e}"


""" The agents in the provided codebase are designed to interact with an environment, perform actions, and learn from their experiences. They are built using various techniques, including reinforcement learning and deep neural networks. Here's a high-level overview of how these agents work:

1. Initialization: Agents are created with specific configurations, such as the type of algorithm they use (e.g., DQN, A2C, PPO), their policy and value function representations (usually deep neural networks), and other hyperparameters.

2. Interaction with the environment: Agents interact with the environment by taking actions based on their current state. They receive observations and rewards from the environment, which they use to update their internal state and knowledge.

3. Action selection: Agents use their policy network to select actions. This can be done using various techniques, such as epsilon-greedy exploration, softmax action selection, or other methods depending on the specific algorithm.

4. Learning: As agents interact with the environment and collect experiences (state, action, reward, next state), they use these experiences to update their policy and value function networks. This is done using techniques like gradient descent, backpropagation, and other optimization methods.

5. Updating agent parameters: Agents update their internal parameters, such as the policy and value function networks, based on the learning process. This allows them to improve their performance over time and adapt to new situations.

6. Resetting episodes: When an episode ends (e.g., the agent reaches a terminal state or a maximum number of steps), the agent resets its internal state and starts a new episode.

The agents in the codebase support various algorithms and can be used for different tasks, such as conversational agents, SQL agents, and more. They can be extended and customized to fit specific use cases and requirements.
TOOLS:
------
Assistant has access to the following tools:'''
FORMAT_INSTRUCTIONS = '''To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```'''
SUFFIX = '''Begin!
Summary:
Prompt.py is a file within the project directory that contains code for creating and managing conversations with users.
 - It includes functions to create prompts, parse user input, and respond accordingly.
 - The file also contains methods for handling errors in user input as well as providing helpful hints when needed.
 - Additionally, it provides support for natural language processing (NLP) tasks such as sentiment analysis and entity extraction. ```

The thought/observation/scratchpad and prefixes are components used by the agent to guide its interaction with various tasks and data sources. Here's a brief explanation of each:

1. Thought: This represents the agent's internal thought process while working on a task. It helps the agent to keep track of its progress and plan its next steps.

2. Observation: This is the result or output obtained after the agent performs an action using a tool. The agent uses this observation to update its knowledge and make further decisions.

3. Scratchpad: The scratchpad is a space where the agent can write down any notes or thoughts while constructing a query or working on a task. It helps the agent to keep track of its progress and organize its thoughts.

4. Prefixes: Prefixes are instructions provided to the agent for specific tasks or interactions. They guide the agent on how to approach a task, what tools to use, and any limitations or rules to follow. Different tasks may have different prefixes to ensure the agent follows the correct approach for each task.

These components help the agent to systematically approach tasks, use appropriate tools, and provide accurate and relevant answers.

 """
