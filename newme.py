import argparse
import os
import sys
import re
import pandas as pd
from typing import List, Tuple
from embedder import CodeExtractor
from chatbot import generate_summary, write_md_files, get_embedding, df_search_sum, chatbot
from utils import split_code_by_TOKEN_MAX_SUMMARY, indexCodebase

MAX_TOKEN_MAX_SUMMARY = 1000

def main():
	parser = argparse.ArgumentParser(description='Code summarization chatbot')
	parser.add_argument('--directory', type=str, default="/ezcoder", help='directory to summarize')
	parser.add_argument('-P', type=str, default="", help='saved db to use ')
	parser.add_argument('--root', type=str, default=f"{os.environ['CODE_EXTRACTOR_DIR']}", help='Where root of project is or env $CODE_EXTRACTOR_DIR')
	parser.add_argument('-n', type=int, default=10, help='number of context chunks to use')
	parser.add_argument('--prompt', type=str, default='What does this code do?', help='gpt prompt')
	parser.add_argument('--chat', type=bool, default=True, help='gpt chat')
	parser.add_argument('--context', type=int, default=10, help='context length')
	parser.add_argument('--max_tokens', type=int, default=1000, help='maximum number of tokens in summary')
	parser.add_argument('--output_dir', type=str, default='', help='directory to save summaries')

	args = parser.parse_args()
	if args.P:
		if not os.path.isfile(args.P):
			parser.error(f"The file specified does not exist.{args.P}")
		df = pd.read_pickle(args.P)
	else:
		if not os.path.isdir(f'{args.root}/{args.directory}'):
			parser.error(f"The directory specified does not exist.{args.root}/{args.directory}")
		if not os.path.isdir(args.root):
			parser.error("The root directory specified does not exist.")
		if not os.path.isdir(args.directory):
			parser.error("The directory specified does not exist.")
		if not isinstance(args.n, int):
			parser.error("The number of context chunks must be an integer.")
		if args.n < 1:
			parser.error("The number of context chunks must be greater than 0.")
		if not isinstance(args.context, int):
			parser.error("The context length must be an integer.")
		if args.context < 1:
			parser.error("The context length must be greater than 0.")
		if not isinstance(args.max_tokens, int):
			parser.error("The maximum number of tokens must be an integer.")
		if args.max_tokens < 1:
			parser.error("The maximum number of tokens must be greater than 0.")
		if len(args.prompt) < 1:
			parser.error("The prompt must be non-empty.")

		extractor = CodeExtractor("{args.root}{args.directory}")
		codebase = extractor.get_files_df()
		df = indexCodebase(codebase, "code")


		df.to_pickle(f"{args.root}/db.pkl")

	if args.chat:
			while True:
				ask = input("\n\033[33mAsk about the files, code, summaries:\033[0m\n\n\033[44mUSER:  \033[0m")
				# q_and_a(df, "What is the code do?", n, 500)# max_tokens * context_n = 15)
				summary_items  = df_search_sum(df, ask, pprint=True, n=n , n_lines=context) 
				chatbot(df, summary_items + prompt, args.context)
	else:
		write_md_files(df, args.output_dir, args.context, args.max_tokens)
if __name__ == '__main__':
	main()