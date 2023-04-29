
import requests
import tiktoken
import pandas as pd
import os
import json
# api_key = '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'
root_dir = '/Users/clockcoin/parsero/'
api_key ="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"
# models = openai.Model.list()

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

def generate_summary_for_directory(directory, df):
    result = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.endswith('.py'):
                file_path = os.path.join(directory, entry.name)
                if df[df['file_path'] == file_path]['summary'].empty:
                    summary = generate_summary_for_file(file_path)
                    df.loc[df['file_path'] == file_path, 'summary'] = summary
                    result[file_path] = summary
                else:
                    result[file_path] = df[df['file_path'] == file_path]['summary'].values[0]
    return result


def get_rel_context_summary(root_dir, df, summary_query):
	df = df[pd.notna(df['summary'])]
	df = df.query(f'summary.str.contains("{summary_query}", regex=False)', engine='python')
	summary_pairs = []
	for _, row in df.iterrows():
			filep = row["file_path"]
			row["file_name"]
			summary = row["summary"]
			summary_pairs.append((filep, summary))
	return summary_pairs

def get_context_code(root_dir, df, code_query):
		df2 = df.query(f'code.str.contains("{code_query}", regex=False)', engine='python')
		code_pairs = []
		for _, row in df2.iterrows():
				filep = row["file_path"]
				row["file_name"]
				with open(root_dir + filep) as file:
						code = f"FILE: {filep}\n"
						for line in file:
								code += line
						code_pairs.append((filep, code))
		return code_pairs


def get_tokens(df = pd.DataFrame):
				EMBEDDING_ENCODING = 'cl100k_base'
				encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
				enc_prompt = encoder.encode(str(prompt))
				tokens = len(encoder.encode(code) + enc_prompt) or 1
				500 + tokens

def generate_table(context_code_pairs, prompt, message="", model="chat-davinci-003-alpha"):
		for filepath, code in context_code_pairs:
					prompt = f"\nSYSTEM: You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up.\nUSER:{code}\nUSER:Please summarize the key features of the specified file within the project directory, and present the information in a concise bullet-point format. Focus on aspects such as the file's content.\nASSISTANT: Sure, here are the key features of the {filepath} file:\n - "
					EMBEDDING_ENCODING = 'cl100k_base'
					encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
					enc_prompt = encoder.encode(str(prompt))
					tokens = len(encoder.encode(code) + enc_prompt) or 1
					base = "https://api.openai.com/v1/completions"
					headers = {
								'Content-Type': 'application/json',
								'Authorization': 'Bearer ' + api_key,
						}
					500 + tokens
					r = requests.post(
							base,
							headers=headers,
							stream=True,
							json={
									"model": model,
									"prompt": prompt,
									"temperature": 0.5,
									"top_p": 0.05,
									"stream": True,
									"n": 1,
									"stop": ["\nSYSTEM:","\nUSER:","\nASSISTANT:" , "\n\n", ],
									"max_tokens": 500 + tokens,
									"presence_penalty": 1,
									"frequency_penalty": 1,
							}
					)


					comp_type = "finish_reason" if model != "chat-davinci-003-alpha" else "finish_details"
					# convert line to utf-8 string
					print(f"\n## FILE : {filepath}" ,  end="\n")
					message += f"\n## FILE : {filepath}"
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
		# df["file_path"] = df["file_path"].str.replace(r"^./", r"")
		# context_code_pairs = get_context_code(root_dir, df, "http")
		# info = generate_table(context_code_pairs, df, "token")
		code_query = 'secu'
		# prompt = "For complexity and relevance columns give the provided code a score from 1 - 10. Use the character '|' as the separator\nYou are an agent operating with other agents to provide information about a codebase or project. You will be passed the full file contents of certain files from the directory that may be relevant to the USER prompt. Return a table row with the columns variable names , function names, imports, exports, summary, importance, complexity, and relevance to user prompt 'What are the exports in the SDK?'. For complexity and relevance columns give the code a score from 1 - 10. Use the character '|' as the separator and end each row with a newline."
		# message = ""
		# last_result= generate_table(context_code_pairs, prompt, message)
