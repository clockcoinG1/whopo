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
		model: str = "code-davinci-002",
		proj_dir: str = "llama",
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
			# model="chat-davinci-003-alpha"
			model="code-davinci-002"
	except NameError:
		# model="chat-davinci-003-alpha"
			model="code-davinci-002"
	comp_type = "finish_reason" if model != "chat-davinci-003-alpha" else "finish_details"
	for _, row in tqdm.tqdm(df.iterrows()):
		time.sleep(3)
		print("sleeping")
		code = row["code"]
		filepath = row["file_path"]
		filename = row["file_name"]
		prompt = f"\nSYSTEM: You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up.\nUSER:{code}\nUSER:Please summarize the key features of the specified file within the project directory, and present the information in a concise bullet-point format. Focus on aspects such as the file's content.\nASSISTANT: Sure, here are the key features of the {filepath} file:\n - "
		encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
		enc_prompt = encoder.encode(str(prompt))
		tokens = len(encoder.encode(code) + enc_prompt) or 1
		# max_token =  abs((4000 +(MAX_TOKEN_COUNT)  - tokens) - 4000)

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
				"stop": ["\nSYSTEM:", "\nUSER:", "\nASSISTANT:","<|im_end|>" ],
				"max_tokens": tokens + 750,
				"presence_penalty": 1,
				"frequency_penalty": 1,
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
							# df.loc[df['file_path'] == filepath, 'summary'] = "NA"
							# print("embedding summaries...\n\n")
							# df['summary_embedding'] = df['summary'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
							
						# else:
						# 	df.loc[df['file_path'] == filepath, 'summary'] = df.loc[df['file_path'] == filepath, 'summary'] + summary.strip()
				# except:
						# print("embedding error")
			
		try:
			old_sum = df[df['file_name'] == filename ]['summary'].values[0]
			df.loc[df['file_name'] == filename, 'summary'] = f'{old_sum}\n{summary.strip()}'
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
							"model": "gpt-3.5-turbo",
							"messages": [
									{"role": "system", "content": f"You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up."},
									{"role": "user", "content": f"{prompt}"}
							],
							"temperature": 2,
							"top_p": 0.05,
							"n": 1,
							"stop": ["<|/im_end|>"],
							"stream": True,
							"max_tokens": int(avail_tokens),
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

if __name__ == '__main__':
		parser = argparse.ArgumentParser(description='Code summarization chatbot')
		parser.add_argument('directory', type=str, help='directory to summarize')
		parser.add_argument('--root', type=str, default='root directory', help='Where root of project is')
		parser.add_argument('-n', type=str, default='Important code', help='context prompt')
		parser.add_argument('-p', type=str, default='What does this code do?', help='gpt prompt')
		parser.add_argument('--context', type=int, default=20, help='context length')
		parser.add_argument('--max_tokens', type=int, default=500, help='maximum number of tokens in summary')

		args = parser.parse_args()
		print(args)
		proj_dir = args.directory
		root_dir = args.root
		prompt = args.p
		n = args.n
		if not os.path.exists(root_dir + "/" + proj_dir):
				print(f"Directory {root_dir + args.directory} does not exist")
				sys.exit()
		context_prompt = args.context

		ce = CodeExtractor(f"{root_dir}{proj_dir}")
		df = ce.get_files_df()
		df = split_code_by_token_count(df, args.max_tokens , "code" )
		df = split_code_by_token_count(df,  col_name="code",  max_tokens=1000)
		df = indexCodebase(df, "code", pickle="safsf.pkl")
		# df = ce.split_code_by_lines(df, max_lines=6)
		df = generate_summary(df)
		rows_with_summary = df[df['summary'] != '' ].dropna()
		# Print file name and summary using the apply function
		# Apply a function to each row of the dataframe
		rows_with_summary.apply(lambda row: print(f"File Name: {row['file_name']}\nSummary: {len(row['summary'])}\n"), axis=1)


		write_md_files(df, os.getcwd() + proj_dir)
		proj_dir_pikl = re.sub(r'[^a-zA-Z]', '', proj_dir)
		df['summary_embedding'] = df['summary'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002') if x else None)
		while True: 
			ask = input("\033[33mAsk about the files, code, summaries:\033[0m\n\033[44mUSER:  \033[0m")
			rez  = df_search_sum(df, ask, pprint=True)
			chatbot(df, rez , 5)
# <|system|> Lets remove redundancies and make this more efficient, SOLID, and pythonic
# ASSISTANT: OK. Here is the improved state of the art code: