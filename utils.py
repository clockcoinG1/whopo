import datetime
import logging
import os
import re
import sys
from glob import glob
from typing import List, Tuple

import openai
import pandas as pd
import tiktoken
from colorlog import ColoredFormatter
from openai.embeddings_utils import get_embedding
from pandas.errors import EmptyDataError

from constants import oai_api_key_embedder, proj_dir

openai.api_key = oai_api_key_embedder
EMBEDDING_ENCODING = 'cl100k_base'
tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)


def setup_logger(name, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create a console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Create a formatter with custom log format and colors
    log_format = "%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s"
    formatter = ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%',
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger(__name__)


def glob_files(directory, extensions):
    all_files = []
    for ext in extensions.split(','):
        all_files.extend(glob(os.path.join(directory, "**", f"*.{ext}"), recursive=True))

    result = []
    for f in all_files:
        file_name = os.path.basename(f)
        file_path = f
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        try:
            with open(f, "r", errors='ignore') as file:
                code = file.read()
        except IsADirectoryError:
            logger.warning(f"Skipping directory: {file_path}")
            continue
        except Exception as e:
            logger.error(f"Error while reading file {file_path}: {e}")
            continue

        result.append({"file_name": file_name, "file_path": file_path, "code": code})

    result = pd.DataFrame(result)
    return result


def split_tokens(tokens: List[str], max_tokens: int) -> List[Tuple[int, int]]:
    token_ranges = []
    start_token = 0
    while start_token < len(tokens):
        end_token = start_token + max_tokens
        token_ranges.append((start_token, end_token))
        start_token = end_token
    return token_ranges

def split_code_by_tokens(df: pd.DataFrame, max_tokens: int = 8100, col_name: str = "code") -> pd.DataFrame:
    def process_row(row: pd.Series) -> List[pd.Series]:
        code = row[col_name]
        tokens = list(tokenizer.encode(code))
        token_ranges = split_tokens(tokens, max_tokens)
        new_rows = []
        for start_token, end_token in token_ranges:
            chunk_tokens = tokens[start_token:end_token]
            chunk_code = "".join(str(token) for token in chunk_tokens)
            new_row = row.copy()
            new_row["code"] = chunk_code
            new_row[f"{col_name}_tokens"] = chunk_tokens
            new_row[f"{col_name}_token_count"] = len(chunk_tokens)
            new_row["file_name"] = f"{new_row['file_name']}_chunk_{start_token}"
            new_rows.append(new_row)
        return new_rows

    new_rows = df.apply(process_row, axis=1).explode().reset_index(drop=True)
    print(f"Created new dataframe\nRows: {new_rows.shape[0]}\nColumns: {new_rows.shape[1]}\n=============================")
    return new_rows

def split_code_by_lines(df: pd.DataFrame, max_lines: int = 1000, col_name: str = "code") -> pd.DataFrame:
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


def indexCodebase(df: pd.DataFrame, col_name: str, pickle: str = "split_codr", code_root: str = "ez11") -> pd.DataFrame:
    try:
        df[f"{col_name}_tokens"] = [list(tokenizer.encode(code)) for code in df[col_name]]
        df[f"{col_name}_total_tokens"] = [len(code) for code in df[f"{col_name}_tokens"]]
        df[f"{col_name}_embedding"] = df[f"{col_name}"].apply(
            lambda x: get_embedding(x, engine='text-embedding-ada-002') if x else None
        )
        logger.info(f"Indexed codebase: {col_name}\t\t{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
        return df
    except EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
        return df
    except Exception as e:
        logger.error(f"Failed to index codebase: {e}")
        return df
    else:
        logger.info("Codebase indexed successfully")
        return df


def write_md_files(df: pd.DataFrame, proj_dir: str = proj_dir) -> None:
    for _, row in df.iterrows():
        header = '# ' + row["file_name"] + '\t\t\t' + row["file_path"] + '\n'
        filepath = row["file_path"]
        filename = row["file_name"]
        summ = row["summary"]
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
            logger.info(f"wrote markdown files: {proj_dir}/docs/{row['file_path'].split('/')[-1]}.md root")