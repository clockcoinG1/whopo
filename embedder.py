import datetime
import os
import pandas as pd
from pandas.errors import EmptyDataError


import uuid
import datetime
import json
import os
import re
from glob import glob
import sys
from typing import List, Optional
import sys

import altair as alt
import openai
import pandas as pd
import requests
import tiktoken
from flask import jsonify
from openai.embeddings_utils import cosine_similarity, get_embedding
import tqdm
from constants import (
	oai_api_key_embedder,
	root_dir,
	proj_dir,
	api_key, 
	EMBEDDING_ENCODING, 
)

openai.api_key = oai_api_key_embedder
tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)

""" root_dir = os.path.expanduser("~")
cwd = os.getcwd()
code_root = f"{root_dir}/parsero"

api_key = "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"
EMBEDDING_ENCODING = 'cl100k_base'

 """

class CodeExtractor:
		def __init__(self, directory):
				self.code_root = code_root
				self.df = pd.DataFrame
				self.EMBEDDING_MODEL = "text-embedding-ada-002"
				self.last_result = ""
				self.max_res = 50
				self.base = 'https://api.openai.com/v1/chat/completions'
				self.headers = {
						'Content-Type': 'application/json',
						'Authorization': 'Bearer ' + api_key,
				}
				self.COMPLETIONS_MODEL = "text-davinci-003"

				self.MAX_SECTION_LEN = 500
				self.SEPARATOR = "<|im_sep|>"
				self.ENCODING = "gpt2"
				# self.pkdf = pd.read_pickle(f"{root_dir}/df2.pkl")
				self.all_files = (
						glob(os.path.join(directory, "**", "*.py"), recursive=True)
						+ glob(os.path.join(directory, "**", "*.ts"), recursive=True)
						+ glob(os.path.join(directory, "**", "*.tsx"), recursive=True)
						+ glob(os.path.join(directory, "**", "*.jsx"), recursive=True)
						+ glob(os.path.join(directory, "**", "*.js"), recursive=True)
				)
				if os.path.isdir(directory):
						self.directory = directory
				else:
						raise ValueError(f"{directory} is not a directory")

		def split_code_by_lines(self, df: pd.DataFrame, max_lines: int = 1) -> pd.DataFrame:
				new_rows = []

				for index, row in df.iterrows():
						code = row["code"]
						lines = code.split("\n")
						lines = list(filter(lambda line: len(line.strip()) > 0, lines))
						lines = list(filter(lambda line: not line.lstrip().startswith("//"), lines))
						lines = list(filter(lambda line: not line.lstrip().startswith("#"), lines))
						line_count = len(lines)

						print(f"Processing file: {row['file_path']}")
						print(f"Original line count: {line_count}")

						if line_count <= max_lines:
								new_rows.append(row)
						else:
								chunks = [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]

								for chunk in chunks:
										new_row = row.copy()
										new_row["code"] = "\n".join(chunk)
										new_rows.append(new_row)

										print(f"Chunk line count: {len(chunk)}")
				self.df = pd.DataFrame(new_rows).reset_index(drop=True)
				return self.df

		def extract_function_name(self, line: str) -> Optional[str]:
				if line is None:
						return None

				if line.startswith("def "):
						return line[len("def ") : line.index("(")].strip()

				
				function_match = re.match(r"\s*(async\s+)?function\s+(\w+)\s*\(", line)
				if function_match:
						return function_match.group(2)

				
				arrow_function_match = re.match(r"\s*const\s+(\w+)\s*=\s*(async\s+)?\(", line)
				if arrow_function_match:
						return arrow_function_match.group(1)

				
				arrow_function_destructuring_match = re.match(
						r"\s*const\s+(\w+)\s*=\s*(async\s+)?\(([^()]+)\)\s*=>",
						line,
				)
				if arrow_function_destructuring_match:
						return arrow_function_destructuring_match.group(1)

				
				tsx_pattern = re.compile(
						r'(?:(?:class|const)\s+(?P<class_name>\w+)(?::\s*\w+)?\s+?=\s+?(?:\w+\s*<.*?>)?\s*?\(\s*?\)\s*?=>\s*?{)|(?:function\s+(?P<function_name>\w+)\s*\(.*?\)\s*?{)',
						re.MULTILINE,
				)

				common_tsx_match = re.match(tsx_pattern, line)
				if common_tsx_match:
						return common_tsx_match.group(1)

				react_function_match = re.match(r"\s*const\s+(\w+)\s*:\s*(\w+)<", line)
				if react_function_match:
						return react_function_match.group(1)
				else:
						return None

		def extract_functions_from_file(self, file_path):
				function_list = []
				last_function_line = None
				function_name = None
				function_body = ""
				with open(file_path, "r") as file:
						for index, line in enumerate(file.readlines()):
								line = line.strip()
								if not line:
										continue

								if function_name is None:
										function_name = self.extract_function_name(line)
										if function_name:
												last_function_line = index

								if function_name and "{" in line:
										if line.count("{") == line.count("}"):
												function_body = "\n".join([function_body, line])
												function_list.append(
														{
																"function_name": function_name,
																"file_path": file_path,
																"code": "".join(open(file_path).readlines()[last_function_line:index]),
														}
												)
												last_function_line = None
												function_name = None
												function_body = ""
										else:
												function_body = "\n".join([function_body, line])
								elif function_name:
										function_body = "\n".join([function_body, line])

				return pd.DataFrame(function_list)

		def extract_functions_from_directory(self, directory_path):
				file_list = []
				for dir_name, subdirs, files in os.walk(directory_path):
						for file in files:
								if file.endswith((".ts", ".tsx", ".js", ".jsx")):
										file_list.append(os.path.join(dir_name, file))

				function_list = []
				for file in file_list:
						function_list.append(self.extract_functions_from_file(file))

				return pd.concat(function_list, ignore_index=True)

		def extract_class_name(self, line: str) -> str:
				if line.startswith("class "):
						return line.split()[1]
				else:
						match = re.match(r"class\s+(\w+)", line)
						return match.group(1) if match else ""

		def extract_code_until_decrease_indent(self, all_lines: list, i: int) -> str:
				current_indent = len(all_lines[i]) - len(all_lines[i].lstrip())
				ret = [all_lines[i]]
				for j in range(i + 1, len(all_lines)):
						line_indent = len(all_lines[j]) - len(all_lines[j].lstrip())
						if line_indent <= current_indent and all_lines[j].strip():
								break
						ret.append(all_lines[j])
				return "\\n".join(ret)

		def extract_code_until_no_space(self, all_lines: list, i: int) -> str:
				ret = [all_lines[i]]
				for j in range(i + 1, i + 10000):
						if j < len(all_lines):
								if len(all_lines[j]) == 0 or all_lines[j][0] in [" ", "\t", ")"]:
										ret.append(all_lines[j])
								else:
										break
				return "\\n".join(ret)

		def get_functions(self, filepath: str) -> dict:
				whole_code = open(filepath).read().replace("\r", "\n")
				all_lines = whole_code.split("\n")
				found = False
				for i, line in enumerate(all_lines):
						if (
								line.startswith("def ")
								or re.match(r"\s*(async\s+)?function\s+(\w+)\s*\(", line)
								or re.match(r"\s*const\s+(\w+)\s*:\s*FunctionComponent", line)
						):
								found = True
								code = self.extract_code_until_no_space(all_lines, i)
								function_name = self.extract_function_name(line)
								if function_name:
										yield {
												"code": code,
												"function_name": function_name,
												"filepath": filepath,
												"line_range": (i, i + len(code.split("\\n")) - 1),
										}
				if not found:
						yield {"code": "", "function_name": "", "filepath": filepath, "line_range": (None, None)}

		def get_classes(self, filepath: str) -> dict:
				whole_code = open(filepath).read().replace("\r", "\n")
				all_lines = whole_code.split("\n")
				for i, line in enumerate(all_lines):
						if line.startswith("class ") or re.match(r"class\s+\w+", line):
								code = self.extract_code_until_no_space(all_lines, i)
								class_name = self.extract_class_name(line)
								line_range = f"{i + 1}-{i + len(code.splitlines())}"
								yield {"code": code, "class_name": class_name, "filepath": filepath, "line_range": line_range}
				if not any(line.startswith("class ") for line in all_lines) and not any(
						re.match(r"class\s+\w+", line) for line in all_lines
				):
						yield {"code": "", "class_name": "", "filepath": filepath, "line_range": ""}

		def get_all_functions(self) -> list:
				return pd.DataFrame(
						{
								"file_name": [os.path.basename(f) for f in self.all_files],
								"file_path": self.all_files,
								"code": [open(f, "r").read() for f in self.all_files],
								"function_name": [open(f, "r").read() for f in self.all_files],
						}
				)

		def get_all_interfaces(self) -> list:
				return [interface for filepath in self.all_files for interface in self.extract_interfaces(filepath)]

		def get_all_classes(self) -> list:
				return [cls for filepath in self.all_files for cls in self.get_classes(filepath)]

		def get_functions_df(self):
				functions = self.get_all_functions()
				df_functions = pd.DataFrame(functions).sort_values(["filepath", "function_name"]).reset_index(drop=True)
				return df_functions

		def get_classes_df(self):
				classes = self.get_all_classes()
				df_classes = pd.DataFrame(classes).sort_values(["filepath", "class_name"]).reset_index(drop=True)
				return df_classes

		def write_to_csv(self):
				df_functions = self.get_functions_df()
				df_functions.to_csv("functions.csv", index=False)

				df_classes = self.get_classes_df()
				df_classes.to_csv("classes.csv", index=False)

				print(f"Wrote {len(df_functions)} functions and {len(df_classes)} classes in {self.directory}")

		def get_files_df(self):
				return pd.DataFrame(
						{
								"file_name": [os.path.basename(f) for f in self.all_files],
								"file_path": self.all_files,
								"code": [open(f, "r").read() for f in self.all_files],
						}
				)



		def indexCodebase(self, df, pickle="split_codr.pkl"):
				try:
						if not os.path.exists(f"{self.code_root}/{pickle}"):
								df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
								df.to_pickle(f"{self.code_root}/{pickle}")
								self.df = df
								print("Indexed codebase: " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
						else:
								self.df = pd.read_pickle(f"{self.code_root}/{pickle}")
				except EmptyDataError as e:
						print(f"Empty data error: {e}")
				except Exception as e:
						print(f"Failed to index codebase: {e}")
				else:
						print("Codebase indexed successfully")



		def add_lines_of_code(self, df: pd.DataFrame) -> pd.DataFrame:
				new_df = (
						df.assign(
								lines_of_code=lambda d: d["code"].str.count("\n").add(1),
								file_path=lambda d: d["file_path"].apply(lambda x: x.replace(os.getcwd(), "")),
						)
						.sort_values("lines_of_code", ascending=False)
						.reset_index(drop=True)
				)
				return new_df


		def extract_tokens(self, df: pd.DataFrame):
				"""
				Extract tokens from code snippets
				"""
				EMBEDDING_ENCODING = 'cl100k_base'
				tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)
				df["tokens"] = [list(tokenizer.encode(code).tokens) for code in df["code"]]
				return df

		def split_code_by_token_count(self, df: pd.DataFrame, max_tokens: int = 8100) -> pd.DataFrame:
				EMBEDDING_ENCODING = 'cl100k_base'
				tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)

				new_rows = []
				for index, row in df.iterrows():
						code = row["code"]
						tokens = row["tokens"]
						token_count = row["token_count"]
						if token_count <= max_tokens:
								new_rows.append(row)
						else:
								
								
								start_token = 0
								while start_token < token_count:
										
										end_token = start_token + max_tokens
										chunk_code = "".join(code[start_token:end_token])
										new_row = row.copy()
										new_row["code"] = chunk_code
										new_row["token_count"] = len(chunk_code.split(" "))
										new_row["file_name"] = f"{new_row['file_name']}_chunk{start_token}"
										new_rows.append(new_row)
										start_token = end_token
				new_df = pd.DataFrame(new_rows)
				print("Created new dataframe")
				print("Rows:", new_df.shape[0])
				print("Columns:", new_df.shape[1], end="\n=============================\n")
				return new_df

		def extract_interfaces(self, filepath: str) -> List[str]:
				interfaces = []

				with open(filepath, "r") as file:
						for line in file:
								interface_match = re.match(r"\s*interface\s+(\w+)\s*(extends\s+\w+)?\s*{", line)
								if interface_match:
										interfaces.append(interface_match.group(1))

				return interfaces

		def df_search(self, df, code_query, n=3, pprint=True, n_lines=7):
				
				result = get_embedding(engine="text-embedding-ada-002", text=code_query)
				embedding = result

				df['similarities'] = df.code_embedding.apply(lambda x: cosine_similarity(x, embedding))

				res = df.sort_values('similarities', ascending=False).head(n)
				res_str = ""
				if pprint:
						for r in res.iterrows():
								print(r[1].file_path + " " + "  score=" + str(round(r[1].similarities, 3)))
								res_str += r[1].file_path + " " + "  score=" + str(round(r[1].similarities, 3))
								print("\n".join(r[1].code.split("\n")[:n_lines]))
								res_str += "\n".join(r[1].code.split("\n")[:n_lines])
								res_str += '-' * 70
								print('-' * 70)


				return res

		def construct_prompt(self, question: str) -> str:
				encoding = tiktoken.get_encoding(self.ENCODING)
				separator_len = len(encoding.encode(self.SEPARATOR))

				relevant_code = self.df_search(df=self.df, code_query=question,n=self.max_res)
				chosen_sections = []
				chosen_sections_len = 0

				for _, row in relevant_code.iterrows():
						code_str = f"Path: {row['file_path']}\nChunked Code:\n{row['code']}"
						code_str_len = len(encoding.encode(code_str))
						if chosen_sections_len + separator_len + code_str_len > self.MAX_SECTION_LEN:
								break

						chosen_sections.append(self.SEPARATOR + code_str)
						chosen_sections_len += separator_len + code_str_len

				
				chosen_sections_str = f"".join(chosen_sections)
				print(f"Selected {len(chosen_sections)} document sections:")
				return f'''<|start_context|>\n Project code to help assistant with answering query "{question}" \n context: {chosen_sections_str}\n<|end_context|>\n<|im_start|>'''

		def summarize_code(self, df):
				res_str = ""
				for r in df.iterrows():
								res_str += f"\n score: {str(round(r[1].similarities, 3))} \t Path : {r[1].file_path} \n"
								n_lines = 5
								res_str += "\n".join(r[1].code.split("\n")[:n_lines])
								
								print('-' * 70)
								print(f" score: {str(round(r[1].similarities, 3))} \t Path : {r[1].file_path} ")
								print('-' * 70)
				return res_str


		def answer_query_with_context(self, query: str, show_prompt: bool = True) -> str:
				COMPLETIONS_API_PARAMS = {
						
						"temperature": 0.85,
						"max_tokens": 3500,
						"model": self.COMPLETIONS_MODEL,
				}
				prompt = self.construct_prompt(
						query,
				)

				if show_prompt:
						print(prompt)

				response = openai.Completion.create(prompt=prompt, **COMPLETIONS_API_PARAMS)
				self.presponse = response["choices"][0]["text"].strip(" \n")
				print("===\n", prompt)
				return response["choices"][0]["text"].strip(" \n")

		def gpt_4(self, prompt="what time is it", context=None,):

				codebaseContext = self.construct_prompt(question=prompt)
				prompt = (
						f"{codebaseContext}\nSYSTEM: You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up."
						f"You always answer truthfully and make sense of the context provided."
						f"USER: {prompt}"
						f"ASSISTANT:"
				)
				EMBEDDING_ENCODING = 'cl100k_base'
				encoder  = tiktoken.get_encoding(EMBEDDING_ENCODING)
				enc_prompt  = encoder.encode(prompt)
				print(f"\033[91mINPUT TOKENS:{str((len(enc_prompt) + len(codebaseContext)))}\033[0m", flush=True)
				print(f"\033[92mAVAILABLE TOKENS:{str((8090 - (len(enc_prompt) + len(codebaseContext))))}\033[0m", end="\n\n")
				print('\x1b[1;37;40m')
				api ="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"
				r = requests.post(
						f"https://api.openai.com/v1/completions",
						headers={
						'Content-Type': 'application/json',
						'Authorization': 'Bearer ' + api,
						},
						json={
								"model": "chat-davinci-003-alpha",
								"prompt": prompt,
								"temperature": 0.8,
								"top_p": 1,
								"n": 1,
								"best_of": 1,
								"stream": True,
								"stop": ["<|im_end|>"],
								"max_tokens": 8090 - (len(enc_prompt) + len(codebaseContext)),
								"presence_penalty": 0,
								"frequency_penalty": 0,
						}
				)

				for line in r.iter_lines():
						message = ""
						if line:
								data = line
								if b"[DONE]" in data:
										print('''
										=========================
										\x1b[0m''')
										break
								else:
											data = json.loads(data[5:])
											if data["object"] == "text_completion":
												if data["choices"][0]["text"]:
													message += data["choices"][0]["text"]
													print(f'{data["choices"][0]["text"]}',flush= True, end="")
											else:
												continue
				
								self.last_result = message

		def chatbot(self, prompt="", brand="Whop"):
				EMBEDDING_ENCODING = 'cl100k_base'
				encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
				enc_prompt  = encoder.encode(prompt)
				codebaseContext = self.construct_prompt(question=prompt)
				cbc_prompt  = encoder.encode(codebaseContext)

				print(f"\033[1;37m{enc_prompt}\t\tTokens:{str(len(enc_prompt) + len(cbc_prompt) )}\033[0m")
				avail_tokens= 3596 - (len(enc_prompt)  + len(cbc_prompt))
				print(f"\n\033[1;37mTOTAL OUTPUT TOKENS AVAILABLE:{avail_tokens}\n\033[0m")
				r = requests.post(
						self.base, headers=self.headers, stream=True,
						json={
								"model": "gpt-3.5-turbo",
								"messages": [
										{"role": "system", "content": f"You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up."},
										{"role": "user", "content": f"\n{codebaseContext}\n"},
										{"role": "user", "content": f"{prompt}"}
								],
								"temperature": 0.8,
								"top_p": 1,
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

def ask(query, gpt=False):
	print(f"{query}\n\nUSER:", flush=False, end="  ")
	question = input()
	if question == "q":
		print("bye")
		exit(0)
	if gpt:
		try:
			extractor.gpt_4(prompt=f"{question}")
			print("\n")
		except:
			print("Sorry, I didn't understand that.")
	else:
		try:
			extractor.chatbot(prompt=f"{question}")
			print("\n")
		except:
			print("Sorry, I didn't understand that.")

def run_chat_loop():
	while True:
		print("QUERY:", flush=False, end="  ")
		question = input()
		ask(input)


""" if len(sys.argv) == 1:
	print( "Must specify a directory as the first argument")
	exit()


 if __name__ == '__main__':

	DIR = sys.argv[1]
	code_root = sys.argv[1] if len(sys.argv) > 2 else "llama"
	gpt_prompt = sys.argv[3] if len(sys.argv) > 3 else "What does this code do?"
	total_context_int = sys.argv[4] if len(sys.argv) > 4 else 20
	token_count = sys.argv[5] if len(sys.argv) > 5 else 200
	context_prompt = sys.argv[6] if len(sys.argv) > 6 else "Important code"
	# code_root = 'llama'
	extractor = CodeExtractor(code_root)
	df = extractor.get_files_df()
	df["tokens"] = [list(tokenizer.encode(code)) for code in df["code"]]
	df["token_count"] = [len(code) for code in df["tokens"]]
	df["token_count"].sum()
	# SPLIT by lines or do .split_by_tokens for more granular / sourcemapped files 
	# df = extractor.split_code_by_lines(df,5)
	df = extractor.split_code_by_token_count(df,200)

	# get token totals total_context_int = 10

	name = f"codebase_pickle-{str(uuid.uuid4())}.pkl"
	extractor.df = df
	extractor.indexCodebase(extractor.df, pickle=name)
	extractor.df = extractor.df_search(df=extractor.df, code_query="ggml",n=10)
	extractor.chatbot("What is Llama LLM ?") """
