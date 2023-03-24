import datetime
import os
import re
from typing import List

import openai
import pandas as pd
import tiktoken
from openai.embeddings_utils import cosine_similarity, get_embedding
from pandas.errors import EmptyDataError
from tqdm import tqdm

from constants import oai_api_key_embedder, proj_dir, root_dir

openai.api_key = oai_api_key_embedder
EMBEDDING_ENCODING = 'cl100k_base'
tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)
import logging
import sys
import tkinter as tk
from tkinter import scrolledtext, ttk

from colorlog import ColoredFormatter



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
				style='%'
		)

		console_handler.setFormatter(formatter)
		logger.addHandler(console_handler)

		return logger



def split_code_by_lines(df: pd.DataFrame, max_lines: int =  1000, col_name: str = "code") -> pd.DataFrame:
    new_rows = []
    df[f"{col_name}_tokens"] = [list(tokenizer.encode(code)) for code in df[col_name]]
    df[f"{col_name}_total_tokens"] = [len(code) for code in df[f"{col_name}_tokens"]]
    for _, row in df.iterrows():
        code = row[col_name]
        lines = [line for line in code.split("\n") if line.strip() and not (line.lstrip().startswith("//") or line.lstrip().startswith("#"))]
        line_count = len(lines)
        print(f"Processing file: {row['file_path']}")
        print(f"Line count: {line_count}")

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
    print("Created new dataframe")
    print("Rows:", new_df.shape[0])
    print("Columns:", new_df.shape[1], end="\n=============================\n")
    return new_df

#tqdm indexing codebase
def indexCodebase(df: pd.DataFrame, col_name: str, pickle: str = "split_codr", code_root: str = "ez11") -> pd.DataFrame:
		"""
		Indexes the codebase and saves it to a pickle file
		
		Args:
		df: pandas dataframe containing the codebase
		col_name: name of the column containing the code
		pickle: name of the pickle file to save the indexed codebase
		
		Returns:
		df: pandas dataframe containing the indexed codebase
		"""
		code_root = root_dir + proj_dir
		try:
				df[f"{col_name}_tokens"] = [list(tokenizer.encode(code)) for code in df[col_name]]
				df[f"{col_name}_total_tokens"] = [len(code) for code in df[f"{col_name}_tokens"]]
				# df.to_pickle(f"{code_root}/{pickle}.pkl")
				df[f"{col_name}_embedding"] = df[f"{col_name}"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002')) 
				print("Indexed codebase: " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
				return df
				# else:
				# 		df = pd.read_pickle(f"{code_root}/{pickle}.pkl")
				# 		return df
		except EmptyDataError as e:
				print(f"Empty data error: {e}")
		except Exception as e:
				print(f"Failed to index codebase: {e}")
		else:
				print("Codebase indexed successfully")
				return df



def split_code_by_tokens(df: pd.DataFrame, max_tokens: int = 8100, col_name: str = "code") -> pd.DataFrame:
		"""
		Splits the code into chunks based on the maximum number of tokens

		Args:
		df: pandas dataframe containing the codebase
		max_tokens: maximum number of tokens allowed in a chunk
		col_name: name of the column containing the code

		Returns:
		new_df: pandas dataframe containing the split code
		"""
		new_rows = []
		for index, row in df.iterrows():
				code = row[col_name]
				tokens = list(tokenizer.encode(code))
				TOKEN_MAX_SUMMARY = len(tokens)
				if TOKEN_MAX_SUMMARY <= max_tokens:
						new_rows.append(row)
				else:
						start_token = 0
						while start_token < TOKEN_MAX_SUMMARY:
								end_token = start_token + max_tokens
								chunk_tokens = tokens[start_token:end_token]
								chunk_code = "".join(str(token) for token in chunk_tokens)
								new_row = row.copy()
								new_row[col_name] = chunk_code
								new_row[f"{col_name}_token_count"] = len(chunk_tokens)
								new_row["file_name"] = f"{new_row['file_name']}_chunk_{start_token}"
								new_rows.append(new_row)
								start_token = end_token

		new_df = pd.DataFrame(new_rows)
		print("Created new dataframe")
		print("Rows:", new_df.shape[0])
		print("Columns:", new_df.shape[1], end="\n=============================\n")
		return new_df

def write_md_files(df: pd.DataFrame, proj_dir: str = proj_dir) -> None:
		"""
		Writes the markdown files
		
		Args:
		df: pandas dataframe containing the codebase
		"""
		for _, row in df.iterrows():
				header = '# ' + row["file_name"] +'\t\t\t' + row["file_path"] + '\n'
				filepath = row["file_path"]
				filename = row["file_name"]
				summ = row["summary"]
				if not os.path.exists(os.path.join(proj_dir, "docs")):
						os.makedirs(os.path.join(proj_dir , "docs"))
				with open(
						os.path.join(
							 proj_dir, "docs", f"{filepath.split('/')[-1]}.md"
						),
						"a",
				) as f:
						summ = re.sub(r"^(Is there anything else.*$|^[\n\s\t].*$|The file is.*$)", "", summ)
						if f.tell() == 0:
								f.write(header)
						f.write(f"# {filename}\n\n")
						f.write(f"## Summary\n\n{summ}\n\n")
						f.write(f"## Code Length\n\n```python\n{len(row['code'])}\n```\n\n")
						f.write(f"## Filepath\n\n```{filepath}```\n\n")
						print(f"\033[1;33;44mwrote markdown files: {proj_dir}/docs/{row['file_path'].split('/')[-1]}.md root\033[0m")