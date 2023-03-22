import numpy as np
from pathlib import Path
import sys
import time
import requests
import argparse
import tiktoken
import pandas as pd
import os
import re
import json
from get_rel_code import api_key
import tqdm
from CodeBaseIndexer import indexCodebase, split_code_by_token_count, write_md_files
from embedder import CodeExtractor
import openai
from openai.embeddings_utils import cosine_similarity, get_embedding
import uuid
from constants import (
	TOKEN_COUNT,
	MAX_TOKEN_COUNT,
	root_dir,
	proj_dir,
	oai_api_key_embedder,
	chat_base,
	base,
	EMBEDDING_ENCODING,
	headers
)

openai.api_key = oai_api_key_embedder
tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)
encoder = tokenizer

def generate_summary(
		df: pd.DataFrame,
		model: str = "chat-davinci-003-alpha",
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
	message = ""
	try:
		if not model:
			model="chat-davinci-003-alpha"
	except NameError:
		model="chat-davinci-003-alph"
			# model="code-davinci-002"
	try:
		if not model:
			model="chat-davinci-003-alpha"
	except NameError:
		model="chat-davinci-003-alpha"	
	comp_type = "finish_reason" if not model or model != "chat-davinci-003-alpha" else "finish_details"
	for _, row in tqdm.tqdm(df.iterrows()):
		code = row["code"]
		filepath = row["file_path"]
		filename = row["file_name"]
		prompt = f"\nSYSTEM: You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up.\nUSER:{code}\nUSER:Please summarize the key features of the specified file within the project directory, and present the information in a concise bullet-point format. Focus on aspects such as the file's content.\nASSISTANT: Sure, here are the key features of the {filepath} code:\n -"
		encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
		enc_prompt = encoder.encode(str(prompt))
		tokens = len(encoder.encode(code) + enc_prompt) or 1
		max_token = abs(7800 - tokens)
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
				"stop": ["\nSYSTEM:", "\nUSER:", "\nASSISTANT:","<|im_end|>" ],
				"max_tokens": 500 + tokens,
				"presence_penalty": 1,
				"frequency_penalty":1,
			}
		)
		summary = ""
		print(f"\n\n\x1b[33m{filepath}\x1b[0m", end='\n', flush=True)
		for line in r.iter_lines():
			data = line.decode('utf-8')
			if data.startswith('data: ') and data != 'data: [DONE]':
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
			old_sum = df[df['file_name'] == filename ]['summary'].values[0]
			df.loc[df['file_name'] == filename, 'summary'] = f'{summary.strip()}'
		except KeyError:
			df.loc[df['file_name'] == filename, 'summary'] = summary.strip()
	return df

def get_tokens(df, colname):

	EMBEDDING_ENCODING = "cl100k_base"
	encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
	code_type = list(df[df.columns[df.columns.to_series().str.contains(colname)]])
	df = ce.df
	for _, row in tqdm.tqdm(ce.df.iterrows()):
		print(row)
		
		filepath = row["file_path"]
		emb_data = "file path: " + filepath + "\n" + str(row[colname])
		tokens = len(encoder.encode(emb_data )) 
		df.loc[df['file_path'] == filepath, 'tokens_summary'] = tokens
	df[['tokens_summary']] = df[['tokens_summary']].applymap(np.int64)
	return df

def df_search(df, summary_query, n=3, pprint=True):
	embedding = get_embedding(engine="text-embedding-ada-002", text=summary_query)
	#df.summary_embedding
	df= df.loc[df.summary_embedding.notnull(), 'summary_embedding']
	embedding = get_embedding(engine="text-embedding-ada-002", text=summary_query)
	df['summary_similarities'] = df.summary_embedding.apply(lambda x: cosine_similarity(x, embedding))
	res = df.sort_values('summary_similarities', ascending=False).head(n)
	res_str = ""
	for r in res.iterrows():
		res_str += f"{r[1].file_path}\n {r[1].summary} \n score={r[1].summary_similarities}"
	return res


def q_and_a(df, question = "What isthe most important file", total = 10, MAX_SECTION_LEN = 7000) -> str:
		SEPARATOR = "<|im_sep|>"
		encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
		separator_len = len(encoder.encode(SEPARATOR))
		relevant_notes = df_search(df, question, total, pprint=True)
		chosen_sections = []
		chosen_sections_len = 0
		for _, row in relevant_notes.iterrows():
				notes_str = f"Path: {row['file_path']}\nSummary:\n{row['summary']}"
				notes_str_len = len(encoder.encode(notes_str))
				if chosen_sections_len + separator_len + notes_str_len > MAX_SECTION_LEN:
						break
				chosen_sections.append(SEPARATOR + notes_str)
				chosen_sections_len += separator_len + notes_str_len
		
		chosen_sections_str = f"".join(chosen_sections)
		print(f"Selected {len(chosen_sections)} document sections:")
		return f'''<|start_context|>\n Project notes to help assistant with answering query "{question}" \n context: {chosen_sections_str}\n<|end_context|>\n<|im_sep|>'''

def chatbot(df, prompt="What does this code do?", n = 4):
			avail_tokens = len(encoder.encode(prompt))
			r = requests.post(
					chat_base, headers=headers, stream=True,
					json={
							"model": "gpt-4-0314",
							"messages": [
									{"role": "system", "content": f"You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up."},
									{"role": "user", "content": f"{prompt}"}
							],
							"temperature": 2,
							"top_p": 0.05,
							"n": 1,
							"stop": ["<|/im_end|>"],
							"stream": True,
							"max_tokens": 8000 - int(avail_tokens),
							"presence_penalty": 0,
							"frequency_penalty": 0,
					}
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


def generate_summary_for_directory(directory, df):
		result = {}
		with os.scandir(directory) as entries:
				for entry in entries:
						if entry.name.endswith(('.py', '.cpp', '.ts', '.js', '.ant')):
								file_path = os.path.join(directory, entry.name)
								if df[df['file_path'] == file_path]['summary'].empty:
										summary = generate_summary_for_file(file_path)
										df.loc[df['file_path'] == file_path, 'summary'] = summary
										result[file_path] = summary
								else:
										result[file_path] = df[df['file_path'] == file_path]['summary'].values[0]
		return result


def df_search_sum(df, summary_query, n=3, pprint=True, n_lines=7):
		embedding  = get_embedding(engine="text-embedding-ada-002", text=summary_query)
		df['summary_simmilarities'] = df.summary_embedding.apply(lambda x: cosine_similarity(x, embedding) if x is not None else 0.8)
		res = df.sort_values('summary_embedding', ascending=False).head(n)
		res_str = ""
		if pprint:
				for r in res.iterrows():
						print(r[1].file_path + " " + "  score=" + str(round(r[1]["summary_simmilarities"], 3)))
						res_str += r[1].file_path + " " + "  score=" + str(round(r[1]["summary_simmilarities"], 3))
						print("\n".join(r[1].summary.split("\n")[:n_lines]))
						res_str += "\n".join(r[1].summary.split("\n")[:n_lines])
						res_str += '-' * 70
						print('-' * 70)
		return res_str


def main():
		parser = argparse.ArgumentParser(description='Code summarization chatbot')
		parser.add_argument('directory', type=str, default="/ezcoder", help='directory to summarize')
		parser.add_argument('--root', type=str, default=f"{os.environ['CODE_EXTRACTOR_DIR']}", help='Where root of project is or env $CODE_EXTRACTOR_DIR')
		parser.add_argument('-n', type=int, default=10, help='number of context chunks to use')
		parser.add_argument('--prompt', type=str, default='What does this code do?', help='gpt prompt')
		parser.add_argument('--chat', type=bool, default=True, help='gpt chat')
		parser.add_argument('--context', type=int, default=10, help='context length')
		parser.add_argument('--max_tokens', type=int, default=1000, help='maximum number of tokens in summary')

		"""     # ======================= # Help-formatting methods # ======================= def format_usage(self): formatter = self._get_formatter() formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups) return formatter.format_help() def format_help(self): formatter = self._get_formatter() # usage formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups) # description formatter.add_text(self.description) # positionals, optionals and user-defined groups for action_group in self._action_groups: formatter.start_section(action_group.title) formatter.add_text(action_group.description) formatter.add_arguments(action_group._group_actions) formatter.end_section() # epilog formatter.add_text(self.epilog) # determine help from format above return formatter.format_help() def _get_formatter(self): return self.formatter_class(prog=self.prog) # ===================== # Help-printing methods # ===================== def print_usage(self, file=None): if file is None: file = _sys.stdout self._print_message(self.format_usage(), file) def print_help(self, file=None): if file is None: file = _sys.stdout self._print_message(self.format_help(), file) def _print_message(self, message, file=None): if message: if file is None: file = _sys.stderr file.write(message) # =============== # Exiting methods # =============== def exit(self, status=0, message=None): if message: self._print_message(message, _sys.stderr) _sys.exit(status) def error(self, message): error(message: string) Prints a usage message incorporating the message to stderr and exits. If you override this in a subclass, it should not return -- it should either exit or raise an exception. """
		args = parser.parse_args()

		if not os.path.isdir(f'{args.root}/{args.directory}'):
			parser.error(f"The directory specified does not exist.{args.root}/{args.directory}")
		# For argparser lets use its  error handling, exit, help and usage formatting and outputting methods from argparse documentation above. Only output code for the main def argparser code for brevity:		if 
		if not os.path.isdir(args.root):
			parser.error("The root directory specified does not exist.")
		if not os.path.isdir(args.directory):
			parser.error("The directory specified does not exist.")
		if not isinstance(args.n, int):
			parser.error("The number of context chunks must be an integer.")
		if  not isinstance(args.context, int):
			parser.error("The context length must be an integer.")
		if not isinstance(args.max_tokens, int):
			parser.error("The maximum number of tokens must be an integer.")
		if not isinstance(args.prompt, str):
			parser.error("The prompt must be a string.")
		if args.n < 1:
			parser.error("The number of context chunks must be greater than 0.")
		if args.context < 1:
			parser.error("The context length must be greater than 0.")
		if args.max_tokens < 1:
			parser.error("The maximum number of tokens must be greater than 0.")
		if len(args.prompt) < 1:
			parser.error("The prompt must be non-empty.")

		print(f"\033[1;32;40m\nSummarizing {args.directory}")
		print(f"\033[1;32;40m\nUsing {args.n} context chunks")
		print(f"\033[1;32;40m\nPrompt: {args.prompt}")

		proj_dir = args.directory.strip() if args.directory is not None else "ez11"
		root_dir = args.root.strip() if args.root is not None else os.getcwd()
		prompt = args.prompt.strip()  if args.prompt is not None else "Explain the code"
		n = args.n if args.n is not None else 20
		context =  args.context if args.context is not None else 15
		max_tokens = args.max_tokens if args.max_tokens is not None else MAX_TOKEN_COUNT
		if not os.path.exists(root_dir + "/" + proj_dir):
				print(f"Directory {root_dir + args.directory} does not exist")
				sys.exit()
		
		ce = CodeExtractor(f"{root_dir}/{proj_dir}")
		df = ce.get_files_df()
		df = split_code_by_token_count(df,  col_name="code",  max_tokens=max_tokens) # OR  df = ce.split_code_by_lines(df, max_lines=6)
		df = indexCodebase(df,"code" , pickle=f"{root_dir}/{proj_dir}.pkl", code_root=f"{root_dir}/{proj_dir}")
		print(f"\033[1;32;40m*" * 20 + "\tGenerating summary...\t" + f"\033[1;32;40m*" * 25)
		df = df[df['code'] != '' ].dropna()
		# df.apply(lambda x: print(x["summary"]), axis=1)
		df = generate_summary(df,  proj_dir=proj_dir)
		df = df[df['summary'] != '' ].dropna()
		print(f"\033[1;32;40m*" * 10 + "\tWriting summary...\t" + f"\033[1;32;40m*" * 10)
		write_md_files(df, f"{proj_dir}".strip('/'))
		proj_dir_pikl = re.sub(r'[^a-zA-Z]', '', f"{root_dir}/{proj_dir}")
		
		print(f"\033[1;34;40m*" * 20 + "\t Embedding summary column ...\t" + f"{root_dir}/{proj_dir}"  + f"\033[1;34;40m*" * 20)
		df['summary_embedding'] = df['summary'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002') if x else None)

		print(f"\033[1;32;40m*" * 40 + "\t Saving embedding summary...\t" + f"{root_dir}/{proj_dir}"  + f"\033[1;32;40m*" * 40)
		df.to_pickle(proj_dir_pikl + '.pkl')
		if args.chat: 
			print(f"\033[1;32;40m*" * 10 + "\t Chat mode \t" + f"{root_dir}/{proj_dir}"  + f"\033[1;32;40m*" * 10)
			while True:
				ask = input("\n\033[33mAsk about the files, code, summaries:\033[0m\n\n\033[44mUSER:  \033[0m")
				# q_and_a(df, "What is the code do?", n, 500)# max_tokens * context_n = 15)
				summary_items  = df_search_sum(df, ask, pprint=True, n=n , n_lines=context) 
				chatbot(df, summary_items , context)

if __name__ == '__main__':
	main()
