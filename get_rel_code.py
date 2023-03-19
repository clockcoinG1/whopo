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
				print(f"<|path: {filep} |>")
				code += (f"\n\`\`\`{filep}\n")
				for x in file:
						code += x
				code += ("\n\`\`\`\n")
		return code




async def generate_table(codebaseContext, prompt, model="chat-davinci-003-alpha"):
		root_dir = '/Users/clockcoin'
		df = pd.read_csv('embedding_2023-03-18-18-34-42.csv')
		code_query = 'SDK'
		codebaseContext = get_context_code(root_dir, df, code_query=code_query)
		prompt = "Return only one row with the variable names , function names, imports, exports, summary, importance, complexity, and notes you may have regarding the file"
		message_temp = f"| variable names | function names | imports | exports | summary | importance | complexity | relevance |\n| -------------- | ------------- | ------- | ------- | ------- | ---------- | ---------- | --------- |\n"
		last_result = ''
		newc = codebaseContext.split("\n\`\`\`\n")
		for codebase_context in newc:
				prompt = (f'{codebase_context}\n{prompt}\n\n\\`\\`\\`\n{message_temp}')
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
								"stop": ["<|im_end|>", "\n\n\n", '\\`\\`\\`', '\n' ],
								"max_tokens": 100,
								"presence_penalty": 1,
								"frequency_penalty": 1,
						}
				)


				message = ""
				comp_type = "finish_reason" if model != "chat-davinci-003-alpha" else "finish_details"
				# convert line to utf-8 string
				for line in r.iter_lines():
								data = line.decode('utf-8')
								if data.startswith('data: ') and data != 'data: [DONE]':
										data = json.loads(data[5:])
										if data["object"] == "text_completion":
												if data["choices"][0][f"{comp_type}"] == "stop":
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
		code_query = 'token'
		codebase_context = get_context_code(root_dir, df, code_query=code_query)
		prompt = "For complexity and relevance columns give the provided code a score from 1 - 10. Use the character '|' as the separator\nYou are an agent operating with other agents to provide information about a codebase or project. You will be passed the full file contents of certain files from the directory that may be relevant to the USER prompt. Return a table row with the columns variable names , function names, imports, exports, summary, importance, complexity, and relevance to user prompt 'What are the exports in the SDK?'. For complexity and relevance columns give the code a score from 1 - 10. Use the character '|' as the separator and end each row with a newline."
		last_result= generate_table(codebase_context, prompt)
async def interact_model(last_result):
  x = await last_result
  print(x)

  for x in last_result:

				print(x)
		with open("info-745.txt", "w") as f:
				f.write(last_result)