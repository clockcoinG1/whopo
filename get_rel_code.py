import openai
import requests
import tiktoken
import pandas as pd
import os
import json
# api_key = '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'
api_key ="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"
# models = openai.Model.list()


# print name , id, and description for each model and human readable time for created

def extract_all_text_files(root_dir: str) -> str:
		text = ''
		for subdir, dirs, files in os.walk(root_dir):
				for file in files:
						if file.endswith(".txt"):
								file_path = subdir + os.path.sep + file
								f = open(file_path, "r")
								for line in f:
										text = text + line
								f.close()
		return text


def get_context_code(root_dir, df, code_query):
		df2 = df.query(f'code.str.contains("{code_query}", regex=False)', engine='python')
		code = ''

		for x in df2.iterrows():
				filep = x[1]["file_path"]
				filen = x[1]["file_name"]
				file = open(root_dir + filep)
				print(f"FILE: {filep}\n")
				code += f"FILE: {filep}\n"
				for x in file:
						code += x
				code += ("\n\`\`\`\n")
		return code, filep


for x in codebaseContext:
		print(len(x[1]))

def generate_table(codebase_context, prompt, model="chat-davinci-003-alpha"):
		newc = codebase_context[0].split("\n\`\`\`\n")
		for code, index in codebaseContext:
					print(index)
					prompt = f"\nSYSTEM: You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up.\nUSER:{codebase_context}\nUSER:Please summarize the key features of the specified file within the project directory, and present the information in a concise bullet-point format. Focus on aspects such as the file's content.\nASSISTANT: Sure, here are the key features of the {codebase_context[1]} file:\n - "
					EMBEDDING_ENCODING = 'cl100k_base'
					encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
					enc_prompt = encoder.encode(str(prompt))
					tokens = len(encoder.encode(codebase_context) + enc_prompt) or 1
					base = "https://api.openai.com/v1/completions"
					headers = {
								'Content-Type': 'application/json',
								'Authorization': 'Bearer ' + api_key,
						}

					r = requests.post(
							base,
							headers=headers,
							stream=True,
							json={
									"model": model,
									"prompt": prompt,
									"temperature": 0.8,
									"top_p": 1,
									"stream": True,
									"n": 1,
									"stop": ["\nSYSTEM:","\nUSER:","\nASSISTANT:" , "\n\n", ],
									"max_tokens": 100 + tokens,
									"presence_penalty": 1,
									"frequency_penalty": 1,
							}
					)


					message = ""
					comp_type = "finish_reason" if model != "chat-davinci-003-alpha" else "finish_details"
					# convert line to utf-8 string
					print(f"\n## FILE : {codebase_context[1]}")
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
																else:
																		message += "\n"
																		print("\n", flush=False, end="")
																		message.strip()
																		continue
		return message


if __name__ == '__main__':
		root_dir = '/Users/clockcoin'
		df = pd.read_csv('embedding_2023-03-18-18-34-42.csv')
		code_query = 'Whop'
		codebaseContext = get_context_code(root_dir, df, code_query=code_query)
		codebaseContext.split("\n\`\`\`\n")
		prompt = "For complexity and relevance columns give the provided code a score from 1 - 10. Use the character '|' as the separator\nYou are an agent operating with other agents to provide information about a codebase or project. You will be passed the full file contents of certain files from the directory that may be relevant to the USER prompt. Return a table row with the columns variable names , function names, imports, exports, summary, importance, complexity, and relevance to user prompt 'What are the exports in the SDK?'. For complexity and relevance columns give the code a score from 1 - 10. Use the character '|' as the separator and end each row with a newline."
		last_result= generate_table(codebaseContext, prompt)

async def test_gen_tb(codebase_context,api_key):
		summaries = await generate_table(codebase_context, api_key)
		for summary in summaries:
			print("File Summary:")
			print(summary)
			print("=" * 80)

asyncio.run(test_gen_tb(codebaseContext, api_key))


import asyncio
from typing import List

async def generate_table(codebase_context: str, api_key: str, model: str = "chat-davinci-003-alpha") -> List[str]:
		sections = codebase_context.split("\n```\n")
		results = await asyncio.gather(*(summarize_section(section, api_key, model) for section in sections))
		return results

async def summarize_section(section: str, api_key: api_key, model: "chat-davinci-003-alpha") -> str:
		prompt = f"\nSYSTEM: You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up.\nUSER:{section}\nUSER:Please summarize the key features of the specified file within the project directory, and present the information in a concise bullet-point format. Focus on aspects such as the file's content, function, dependencies, and any other relevant details that can provide a clear understanding of its purpose and importance in the project.\n"

		EMBEDDING_ENCODING = 'cl100k_base'
		encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
		enc_prompt = encoder.encode(prompt)
		tokens = len(encoder.encode(section) + enc_prompt) or 1

		url = "https://api.openai.com/v1/completions"
		headers = {
				'Content-Type': 'application/json',
				'Authorization': f'Bearer {api_key}',
		}

		response = requests.post(
				url,
				headers=headers,
				stream=True,
				json={
						"model": model,
						"prompt": prompt,
						"temperature": 0.8,
						"top_p": 1,
						"stream": True,
						"n": 1,
						"stop": ["\nSYSTEM:", ],
						"max_tokens": 100 + tokens,
						"presence_penalty": 1,
						"frequency_penalty": 1,
				}
		)

		message = ""
		comp_type = "finish_reason" if model != "chat-davinci-003-alpha" else "finish_details"

		for line in response.iter_lines():
				data = line.decode('utf-8')
				if data.startswith('data: ') and data != 'data: [DONE]':
						data = json.loads(data[5:])
						if data["object"] == "text_completion":
								choice = data["choices"][0][comp_type]
								if choice == "stop":
										break
								else:
										message += data["choices"][0]["text"]

		return message.strip()
