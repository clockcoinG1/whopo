import json
import os
import re
from glob import glob
from typing import List, Optional

import altair as alt
import openai
import pandas as pd
import requests
import tiktoken
from flask import jsonify
from openai.embeddings_utils import cosine_similarity, get_embedding

# openai.api_key = "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"
root_dir = os.path.expanduser("~")
cwd = os.getcwd()
code_root = f"{root_dir}/Downloads/whopt"
api_key = "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"
openai.api_key = api_key


class CodeExtractor:
		"""
		Extracts functions and classes from TypeScript/JavaScript/Python files in a directory.
		"""
		def __init__(self, directory):
				self.EMBEDDING_MODEL = "text-embedding-ada-002"
				self.base = 'https://api.openai.com/v1/chat/completions'
				self.COMPLETIONS_MODEL = "text-davinci-003"
				self.headers = {
						'Content-Type': 'application/json',
						'Authorization': 'Bearer ' + api_key,
				}

				self.MAX_SECTION_LEN = 1500
				self.SEPARATOR = "\n* "
				self.ENCODING = "gpt2"
				self.df = pd.DataFrame()
				self.all_files = glob(os.path.join(directory, "**", "*.py"), recursive=True) + \
													glob(os.path.join(directory, "**", "*.ts"), recursive=True) + \
													glob(os.path.join(directory, "**", "*.tsx"), recursive=True) + \
													glob(os.path.join(directory, "**", "*.jsx"), recursive=True) + \
													glob(os.path.join(directory, "**", "*.js"), recursive=True)
				if os.path.isdir(directory):
						self.directory = directory
				else:
						raise ValueError(f"{directory} is not a directory")

		def split_code_by_lines(self, df: pd.DataFrame, max_lines: int = 50) -> pd.DataFrame:
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
								chunks = [lines[i:i + max_lines] for i in range(0, len(lines), max_lines)]

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
					return line[len("def "): line.index("(")].strip()

			# Standard and async functions
			function_match = re.match(r"\s*(async\s+)?function\s+(\w+)\s*\(", line)
			if function_match:
					return function_match.group(2)

			# TypeScript arrow functions
			arrow_function_match = re.match(r"\s*const\s+(\w+)\s*=\s*(async\s+)?\(", line)
			if arrow_function_match:
					return arrow_function_match.group(1)

			# TypeScript arrow functions with destructuring and type annotations
			arrow_function_destructuring_match = re.match(
					r"\s*const\s+(\w+)\s*=\s*(async\s+)?\(([^()]+)\)\s*=>",
					line,
			)
			if arrow_function_destructuring_match:
					return arrow_function_destructuring_match.group(1)

			# React FunctionComponent and other generics
			tsx_pattern = re.compile(r'(?:(?:class|const)\s+(?P<class_name>\w+)(?::\s*\w+)?\s+?=\s+?(?:\w+\s*<.*?>)?\s*?\(\s*?\)\s*?=>\s*?{)|(?:function\s+(?P<function_name>\w+)\s*\(.*?\)\s*?{)', re.MULTILINE)

			common_tsx_match = re.match(tsx_pattern, line)
			if common_tsx_match:
					return common_tsx_match.group(1)

			react_function_match = re.match(r"\s*const\s+(\w+)\s*:\s*(\w+)<", line)
			if react_function_match:
					return react_function_match.group(1)
			else:
				return None

		def extract_functions_from_file(self,file_path):
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
																"code": "".join(
																		open(file_path).readlines()[last_function_line:index]
																),
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
			if not any(line.startswith("class ") for line in all_lines) and not any(re.match(r"class\s+\w+", line) for line in all_lines):
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
			return [
					interface
					for filepath in self.all_files
					for interface in self.extract_interfaces(filepath)
			]
		def get_all_classes(self) -> list:
				return [
						cls
						for filepath in self.all_files
						for cls in self.get_classes(filepath)
				]

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

		def indexCodebase(self, df):
			try:
				all_funcs = self.get_files_df()
				all_funcs = self.add_lines_of_code(all_funcs)
				if not os.path.exists(f"{root_dir}/df.pkl"):
					df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
					df.to_pickle(f"{root_dir}/df.pkl")
					self.df = df
				df = pd.read_pickle(f"{root_dir}/df.pkl")
				df['file_path'] = df['file_path'].apply(lambda x: x.replace(root_dir, ""))
				df.to_csv("embedding.csv", index=False)
				self.df = df
				df.head()
			except Exception:
				print("Failed to index codebase")

		def add_lines_of_code(self, df: pd.DataFrame) -> pd.DataFrame:
				new_df = (
						df.assign(
								lines_of_code=lambda d: d["code"].str.count("\n").add(1),
								file_path=lambda d: d["file_path"].apply(
										lambda x: x.replace(os.getcwd(), "")
								),
						)
						.sort_values("lines_of_code", ascending=False)
						.reset_index(drop=True)
				)
				return new_df


		def split_code_by_token_count(self, df: pd.DataFrame, max_tokens: int = 2092) -> pd.DataFrame:
				"""Use the same tokenizer as the pre-trained model """
				EMBEDDING_ENCODING = 'cl100k_base'
				tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)
				new_rows = []

				# Iterate over all rows in the dataframe
				for index, row in df.iterrows():
						code = row["code"]
						# Tokenize the code
						tokens = list(tokenizer.encode(code))
						# Get the number of tokens in the code
						token_count = len(tokens)

						if token_count <= max_tokens:
								new_rows.append(row)
						else:
								# Get the halfway point of the code
								halfway_point = token_count // 2
								# Find the first occurrence of two newline characters in the code
								split_point = None

								# We have added the verbose debugging to see the value of some variables while running the code.
								# In each iteration, we print the token count, i.e., how many tokens we have in the code.
								# If the token count is greater than max_tokens, we will try to split the code into two parts.
								# The goal is to create two sets of codes with as few a number of tokens as possible but not exceeding the max_tokens limit.
								print(f"Token count: {token_count}")
								for i in range(halfway_point, len(tokens) - 1):
										if tokens[i].value == "\n" and tokens[i + 1].value == "\n":
												split_point = i
												break
								# If we found a place to split the code, then we create two new rows with the first set of codes and the second set of codes, respectively.
								if split_point:
										# Split the code into two parts
										first_half = "".join(token.value for token in tokens[: split_point + 1])
										second_half = "".join(token.value for token in tokens[split_point + 1:])

										# Create a new row for the first half
										new_row1 = row.copy()
										new_row1["code"] = first_half
										new_rows.append(new_row1)

										# Create a new row for the second half
										new_row2 = row.copy()
										new_row2["code"] = second_half
										new_rows.append(new_row2)
								# If we are unable to split the code, then we keep the code as it is.
								else:
										new_rows.append(row)

				return pd.DataFrame(new_rows).reset_index(drop=True)


		def extract_interfaces(self, filepath: str) -> List[str]:
				interfaces = []

				with open(filepath, "r") as file:
						for line in file:
								interface_match = re.match(r"\s*interface\s+(\w+)\s*(extends\s+\w+)?\s*{", line)
								if interface_match:
										interfaces.append(interface_match.group(1))

				return interfaces

		def extract_interface(self, line: str) -> Optional[str]:
				if line.startswith("interface "):
						intf = " ".join(line.split(" ")[1:]).strip()
						return intf

		def write_to_csv(self):
				df_functions = self.get_functions_df()
				df_functions_split = self.split_code_by_token_count(df_functions)
				df_functions_split.to_csv("functions.csv", index=False)

				df_classes = self.get_classes_df()
				df_classes_split = self.split_code_by_token_count(df_classes)
				df_classes_split.to_csv("classes.csv", index=False)

				print(f"Wrote {len(df_functions_split)} functions and {len(df_classes_split)} classes in {self.directory}")

		def search_functions(self, df, code_query, n=3, pprint=True, n_lines=7):
				# Replace the `get_embedding` function with the GPT-4 embeddings function call
				result = get_embedding(
						engine="text-embedding-ada-002",
						text=code_query
				)
				embedding = result

				df['similarities'] = df.code_embedding.apply(lambda x: cosine_similarity(x, embedding))

				res = df.sort_values('similarities', ascending=False).head(n)
				res_str = ""
				if pprint:
						for r in res.iterrows():
								print(r[1].file_path + " " + "  score=" + str(round(r[1].similarities, 3)))
								res_str += (r[1].file_path + " " + "  score=" + str(round(r[1].similarities, 3)))
								print("\n".join(r[1].code.split("\n")[:n_lines]))
								res_str +=  "\n".join(r[1].code.split("\n")[:n_lines])
								res_str += ('-' * 70)
								print('-' * 70)
				return res

		def construct_prompt(self, question: str) -> str:
			encoding = tiktoken.get_encoding(self.ENCODING)
			separator_len = len(encoding.encode(self.SEPARATOR))

			relevant_code = self.search_functions(df=self.df, code_query=question, n=5)
			chosen_sections = []
			chosen_sections_len = 0

			for _, row in relevant_code.iterrows():
					code_str = f"File: {row['file_name']}\n" \
										f"Path: {row['file_path']}\n" \
										f"Similarity: {row['similarities']}\n" \
										f"Lines of Code: {row['lines_of_code']}\n" \
										f"Code:\n{row['code']}\n"

					code_str_len = len(encoding.encode(code_str))

					if chosen_sections_len + separator_len + code_str_len > self.MAX_SECTION_LEN:
							break

					chosen_sections.append(self.SEPARATOR + code_str)
					chosen_sections_len += separator_len + code_str_len

			# Useful diagnostic information
			print(f"Selected {len(chosen_sections)} document sections:")
			chosen_sections_str = "".join(chosen_sections)
			print(chosen_sections_str)

			return f'\nCode base context from embeddings dataframe:\n\n{chosen_sections_str}<|/knowledge|>'



		def answer_query_with_context(
				self,
				query: str,
				show_prompt: bool = True
		) -> str:
				COMPLETIONS_API_PARAMS = {
						# We use temperature of 0.0 because it gives the most predictable, factual answer.
						"temperature": 0.85,
						"max_tokens": 200,
						"model": self.COMPLETIONS_MODEL,
				}
				prompt = self.construct_prompt(
						query,
				)

				if show_prompt:
						print(prompt)

				response = openai.Completion.create(
										prompt=prompt,
										**COMPLETIONS_API_PARAMS
								)
				self.presponse = response["choices"][0]["text"].strip(" \n")
				print("===\n", prompt)
				return response["choices"][0]["text"].strip(" \n")


		def chatbot(self, prompt="", brand = "Whop"):
					messages = []
					codebaseContext = self.construct_prompt(question = prompt)
		 			print(codebaseContext)
					r = requests.post(
										self.base, headers=self.headers , stream=True,
										json={
												"model": "gpt-3.5-turbo",
												"messages": [
													{"role": "system", "content": f"You are ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up." },
													{"role": "user", "content": f"\n{codebaseContext}\n"},
													{ "role": "user", "content": f"USER:{prompt}\nASSISTANT:" }
											],
												"temperature": 0.8,
												"top_p": 1,
												"n": 1,
												"stop": ["\nUSER:"],
												"stream": True,
												"max_tokens": 2100,
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
									# output += str(data)
									data = json.loads(data[5:])
									if data["object"] == "chat.completion.chunk":
												if data["choices"][0]["finish_reason"] == "stop":
																				break
												else:
																				if "content" in data["choices"][0]["delta"]:
																					message += data["choices"][0]["delta"]["content"]
																					print(data["choices"][0]["delta"]["content"], flush=True, end="")
																				else:
																					message += " "



extractor = CodeExtractor(f"{root_dir}/Downloads/whopt")
df = extractor.get_files_df()
df = extractor.split_code_by_lines(df, 5)
df['file_path'] = df['file_path'].apply(lambda x: x.replace(code_root, ""))
df = extractor.add_lines_of_code(df)
# df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df.to_csv(f"{root_dir}/df_1.csv", index=False, header=True)
extractor.indexCodebase(df)
df = extractor.df
df = pd.read_pickle(f"{root_dir}/df.pkl")
prompt = extractor.search_functions(df,  "SDK features", 10)


selected_columns = ["code", "file_path", "file_name"]
result_string = prompt[selected_columns].to_string(index=False, justify="left", header=False, show_dimensions=True)
result_string = re.sub(r'\s+', ' ', result_string, flags=re.MULTILINE)

len(result_string)
prompt.to_string()

extractor.chatbot(prompt=f"\nUSER: List the features\n")


df = pd.read_csv(f"df.csv")
# now for each row, make a completions api call for each function in turn, and sum the logits scores returne
# extractor.split_code_by_token_count(df, 400)
# if os.path.exists("all_functions.pkl"):
df = pd.read_pickle("all_functions1.pkl")
extractor.extract_functions_from_file("/Users/clockcoin/Downloads/whopt/lib/get-user-sdk/pages.ts")
extractor.extract_functions_from_directory("/Users/clockcoin/Downloads/whopt")
extractor.extract_interfaces("/Users/clockcoin/Downloads/whopt/lib/get-user-sdk/pages.ts")
df = extractor.extract_interfaces("/Users/clockcoin/Downloads/whopt/pages/index.tsx")
extractor.get_all_interfaces()


df = extractor.extract_functions_from_directory(f"{root_dir}/Downloads/whopt")

def filter_functions_by_keyword(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
		"""
		Filters the functions in the given data frame by searching for the keyword in the function name.
		"""
		return df[df["code"].str.contains("id", case=False, regex=False, na=False)]


def filter_functions_by_path(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
		"""
		Filters the functions in the given data frame by searching for the keyword in the function name.
		"""
		return df[df["file_path"].str.contains("lib", case=False, regex=False, na=False)]

filter_functions_by_keyword(df, "token")
filter_functions_by_path(df, "dist")
