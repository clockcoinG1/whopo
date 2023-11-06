import logging
import os
from glob import glob
import sys
from colorlog import TTYColoredFormatter
import pandas as pd
from typing import List
import re
import openai
import tiktoken

from df_logger import setup_logger


openai.api_key = os.environ["OPENAI_API_KEY"]
EMBEDDING_ENCODING = "cl100k_base"
tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)


def glob_files(directory, extensions):
    logger = setup_logger("index_codebase", logging.DEBUG)
    logger.info(f"Indexing files in {directory} with extensions {extensions}")
    all_files = []
    for ext in extensions.split(","):
        all_files.extend(glob(os.path.join(directory, "**", f"*.{ext}"), recursive=True))

    result = []
    for f in all_files:
        file_name = os.path.basename(f)
        file_path = f
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        try:
            with open(f, "r", errors="ignore") as file:
                code = file.read()
        except IsADirectoryError:
            logger.warning(f"Skipping directory: {f}")
            continue
        except Exception:
            logger.warning(f"Skipping file: {f}")
            continue

        result.append({"file_name": file_name, "file_path": file_path, "code": code})

    result = pd.DataFrame(result)
    logger.info(f"Indexed {len(result)} files")
    return result


def split_code_by_tokens(df: pd.DataFrame, max_tokens: int = 8100, col_name: str = "code") -> pd.DataFrame:
    logger = setup_logger("index_codebase")

    def tokenize_code(code: str) -> List[int]:
        logger.info(f"Tokenizing code: {len(code)}")
        return list(tokenizer.encode(code))

    def split_tokens(tokens: List[int], max_tokens: int) -> List[List[int]]:
        logger.info(
            f"Splitting tokens: {len(tokens) } into chunks of {max_tokens} tokens and"
            f" {len(tokens) % max_tokens} remainder"
        )
        return [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]

    def create_new_row(row: pd.Series, chunk_tokens: List[int], start_token: int) -> pd.Series:
        new_row = row.copy()
        chunk_code = "".join(str(token) for token in chunk_tokens)
        new_row[col_name] = chunk_code
        new_row[f"{col_name}_tokens"] = chunk_tokens
        new_row[f"{col_name}_token_count"] = len(chunk_tokens)
        new_row["file_name"] = f"{new_row['file_name']}_chunk_{start_token}"
        new_row["file_path"] = f"{new_row['file_path']}_chunk_{start_token}"
        return new_row

    new_rows = []
    for index, row in df.iterrows():
        code = row[col_name]
        tokens = tokenize_code(code)

        if len(tokens) <= max_tokens:
            new_rows.append(row)
        else:
            chunks = split_tokens(code, max_tokens)
            logger.debug(f"Splitting code into {len(chunks)} chunks")
            new_rows.extend(
                create_new_row(row, chunk_tokens, start_token) for start_token, chunk_tokens in enumerate(chunks)
            )

    new_df = pd.DataFrame(new_rows)
    logger.info(f"New DF has {len(new_df)} rows with {new_df[col_name].str.len().sum()} tokens")
    return new_df


def index_codebase(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """This code indexes the codebase by tokenizing the code and computing the code embeddings."""
    logger = setup_logger("Indexing codebase")
    try:
        df[f"{col_name}_tokens"] = [list(tokenizer.encode(code)) for code in df[col_name]]
        df[f"{col_name}_total_tokens"] = [len(code) for code in df[f"{col_name}_tokens"]]
        df[f"{col_name}_embedding"] = df[f"{col_name}"].apply(
            lambda x: openai.embeddings.create(input=x if x is not None else float(0), model='text-embedding-ada-002')
            .data[0]
            .embedding
        )
        return df
    except Exception as e:
        logger.error(f"Error indexing codebase: {e}")
        raise "Error indexing codebase"
    finally:
        logger.info(f"Indexed {len(df)} rows")


def split_code_by_lines(df: pd.DataFrame, max_lines: int = 1000, col_name: str = "code") -> pd.DataFrame:
    logger = setup_logger("Splitting code by lines")
    new_rows = []
    df[f"{col_name}_tokens"] = [list(tokenizer.encode(code)) for code in df[col_name]]
    df[f"{col_name}_total_tokens"] = [len(code) for code in df[f"{col_name}_tokens"]]
    for _, row in df.iterrows():
        code = row[col_name]
        lines = [
            line
            for line in code.split("\n")
            if line.strip() and not (line.lstrip().startswith("//") or line.lstrip().startswith("#"))
        ]
        line_count = len(lines)
        if line_count <= max_lines:
            new_rows.append(row)
        else:
            chunks = [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]
            for index, chunk in enumerate(chunks):
                new_row = row.copy()
                new_row[col_name] = "\n".join(chunk)
                new_row["file_name"] = f"{new_row['file_name']}_chunk_{index * max_lines}-{(index + 1) * max_lines}"
                new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    logger.info("Created new dataframe")
    logger.info(f"Rows: {new_df.shape[0]}")
    logger.info(f"Columns: {new_df.shape[1]}\n=============================")
    return new_df


def write_md_files(df: pd.DataFrame, proj_dir: str = "langchain") -> None:
    logger = setup_logger("Writing MDfiles")
    for _, row in df.iterrows():
        filepath = row["file_path"]
        filename = row["file_name"]
        summ = row["summary"]
        logger.info(f"Writing {filename} files to {filepath} directory")
        header = f"# {filename}\t\t\t{filepath}\n"
        if not os.path.exists(os.path.join(proj_dir, "docs")):
            os.makedirs(os.path.join(proj_dir, "docs"))
        with open(
            os.path.join(proj_dir, "docs", f"{filepath.split('/')[-1]}.md"),
            "a",
        ) as f:
            summ = re.sub(r"^(Is there anything else.*$|^[\n\s\t].*$|The file is.*$)", "", summ)
            if f.tell() == 0:
                f.write(header)
            f.write(f"# {filename}\n\n")
            f.write(f"## Summary\n\n{summ}\n\n")
            f.write(f"## Code Length\n\n```python\n{len(row['code'])}\n```\n\n")
            f.write(f"## Filepath\n\n```{filepath}```\n\n")


""" In the context of software development and artificial intelligence, an agent is an autonomous entity that can perceive its environment, make decisions, and take actions to achieve specific goals. Agents can be simple or complex, depending on the problem they are designed to solve. Here's a general overview of how agents work:

1. **Perception**: Agents receive input from their environment through sensors or other data sources. This input can be in the form of raw data, such as images, audio, or text, or it can be pre-processed information, such as feature vectors or structured data.

2. **Processing**: Once the agent has perceived its environment, it processes the input data to make sense of it. This can involve various techniques, such as pattern recognition, machine learning, or rule-based systems, to extract meaningful information from the input data.

3. **Decision-making**: Based on the processed information, the agent makes decisions about what actions to take. This can involve selecting the best action from a set of possible actions, or it can involve more complex planning and reasoning processes to determine the optimal sequence of actions to achieve a goal.

4. **Action**: Once the agent has decided on an action, it executes the action in its environment. This can involve sending commands to actuators, such as motors or other hardware devices, or it can involve updating internal data structures or communicating with other agents or systems.

5. **Learning**: In many cases, agents can learn from their experiences and improve their performance over time. This can involve updating their internal models, adjusting their decision-making processes, or refining their perception and processing techniques.

Agents can be implemented using various programming languages, frameworks, and libraries, depending on the specific requirements of the problem they are designed to solve. Some popular agent-based frameworks include the Belief-Desire-Intention (BDI) architecture, the Cognitive Agent Architecture (Cougaar), and the Multi-Agent System (MAS) framework.
 """
