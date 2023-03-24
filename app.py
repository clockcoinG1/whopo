import argparse
import os
import sys
import pandas as pd
import re
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path
from utils import split_code_by_lines, split_code_by_tokens, setup_logger
from chatbot import indexCodebase, df_search_sum, generate_summary, write_md_files, chat_interface
from glob_files import glob_files
from openai.embeddings_utils import get_embedding


def process_arguments():
		parser = argparse.ArgumentParser(description='Code summarization chatbot')
		parser.add_argument('--directory', type=str, default=os.getcwd(), help='directory to summarize')
		parser.add_argument('-P', type=str, default="", help='saved db to use')
		parser.add_argument('--root', type=str, default=f"{os.environ['CODE_EXTRACTOR_DIR']}", help='Where root of project is or env $CODE_EXTRACTOR_DIR')
		parser.add_argument('-n', type=int, default=10, help='number of context chunks to use')
		parser.add_argument('--prompt', type=str, default='What does this code do?', help='gpt prompt')
		parser.add_argument('--chat', type=bool, default=True, help='gpt chat')
		parser.add_argument('--context', type=int, default=10, help='context length')
		parser.add_argument('--max_tokens', type=int, default=1000, help='maximum number of tokens in summary')
		parser.add_argument('--ext', type=str, default="py,ts,js,md,txt", help='file ext to target')
		parser.add_argument('--split_by', type=str, choices="tokens,lines", default='lines', help='split code by tokens or lines')
		return parser.parse_args()


def validate_arguments(args):
		if args.P:
				if not os.path.isfile(args.P):
						raise ValueError(f"The file specified does not exist: {args.P}")

		if not os.path.exists(args.root  + '/' + args.directory.strip()):
				raise ValueError(f"Directory {args.root  + '/' + args.directory.strip()} does not exist")




def main():
	logger = setup_logger("SOTA_Logger")
	try:
				args = process_arguments()
				validate_arguments(args)
				proj_dir = Path(args.root.strip(), args.directory.strip())
				root_dir = Path(args.root.strip())
				prompt = args.prompt.strip()
				ext = args.ext
				n = args.n
				context = args.context
				max_tokens = args.max_tokens
				split_by = args.split_by
				if args.P:
						df = pd.read_pickle(args.P)
						chat_interface(df, n, context)

				else:
						logger.info(f"Summarizing {args.directory}\nUsing {args.n} context chunks\nPrompt: {args.prompt}")
						df = glob_files(str(proj_dir), ext)
						if split_by == 'lines':
								df = split_code_by_lines(df, max_lines=25)
						else:
								df = split_code_by_tokens(df, max_tokens=max_tokens)
						df = indexCodebase(df, "code")
						logger.info("Generating summary...")
						df = df[df['code'] != ''].dropna()
						df = generate_summary(df, proj_dir=str(proj_dir))
						df = df[df['summary'] != ''].dropna()
						logger.info("Writing summary...")
						write_md_files(df, str(proj_dir).strip('/'))

						proj_dir_pikl = re.sub(r'[^a-zA-Z]', '', f"{root_dir}/{proj_dir}")
						print(f"\033[1;34;40m*" * 20 + "\t Embedding summary column ...\t" + f"{root_dir}/{proj_dir}"  + f"\033[1;34;40m*" * 20)
						df['summary_embedding'] = df['summary'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002') if x else None)
						print(f"\033[1;32;40m*" * 40 + "\t Saving embedding summary...\t" + f"{root_dir}/{proj_dir}")
						chat_interface(df,n,context)
						chat_interface(df,5,5)

	except ValueError as e:
			print(f"Error: {e}")
			sys.exit(1)
	except Exception as e:
			print(f"Unexpected error: {e}")
			sys.exit(1)

if __name__ == "__main__":
		main()