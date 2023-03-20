import numpy as np
from pathlib import Path
import openai
import requests
import tiktoken
import pandas as pd
import os
import json
from get_rel_code import api_key, root_dir
import tqdm
from openai.embeddings_utils import cosine_similarity, get_embedding
openai.api_key = "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"
base = "https://api.openai.com/v1/completions"
chat_base = 'https://api.openai.com/v1/chat/completions'
headers = {
	'Content-Type': 'application/json',
	'Authorization': 'Bearer ' + api_key,
}
def generate_summary(context_code_pairs, df,model="chat-davinci-003-alpha"):
	message = ""
	try:
		if not model:
			model="chat-davinci-003-alpha"
	except NameError:
		model="chat-davinci-003-alpha"
	
	comp_type = "finish_reason" if not model or model != "chat-davinci-003-alpha" else "finish_details"
	for filepath, code in context_code_pairs:
		filepath  = filepath.replace("/Downloads/whopt","")
		prompt = f"\nSYSTEM: You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up.\nUSER:{code}\nUSER:Please summarize the key features of the specified file within the project directory, and present the information in a concise bullet-point format. Focus on aspects such as the file's content.\nASSISTANT: Sure, here are the key features of the {filepath} file:\n - "
		EMBEDDING_ENCODING = 'cl100k_base'
		encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
		enc_prompt = encoder.encode(str(prompt))
		tokens = len(encoder.encode(code) + enc_prompt) or 1
		max_token = 500 + tokens
		r = requests.post(
			base,
			headers=headers,
			stream=True,
			json={
				"model": model,
				"prompt": prompt,
				"temperature": 2,
				"top_p": 0.05,
				"stream": True,
				"n": 1,
				"logit_bias": {
					[27, 91, 320, 62, 437, 91, 29, 198] : -100
					},
				"stop": ["\nSYSTEM:", "\nUSER:", "\nASSISTANT:", "\n\n", ],
				"max_tokens": 500 + tokens,
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
		
		df.loc[df['file_path'] == filepath, 'summary'] = summary.strip()
	return message

generate_summary([df ])



def get_tokens(df, colname):
	EMBEDDING_ENCODING = "cl100k_base"
	encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
	colname = "summary"
	code_type = list(df[df.columns[df.columns.to_series().str.contains(colname)]])
	for _, row in tqdm.tqdm(df.iterrows()):
		
		filepath = row["file_path"]
		emb_data = "file path: " + filepath + "\n" + row["summary"]
		tokens = len(encoder.encode(emb_data )) 
		df.loc[df['file_path'] == filepath, 'tokens_summary'] = tokens
	df[['tokens_summary']] = df[['tokens_summary']].applymap(np.int64)
	return df

def df_search(df, summary_query, n=3, pprint=True):
	embedding = get_embedding(engine="text-embedding-ada-002", text=summary_query)
	df['summary_similarities'] = df.summary_embedding.apply(lambda x: cosine_similarity(x, embedding))
	res = df.sort_values('summary_similarities', ascending=False).head(n)
	res_str = ""
	for r in res.iterrows():
		
		res_str += f"{r[1].file_path}\n {r[1].summary} \n score={r[1].summary_similarities}"
		
	return res




def q_and_a(question = "What isthe most important file", total = 10, MAX_SECTION_LEN = 7000) -> str:
		SEPARATOR = "<|im_sep|>"
		encoding = tiktoken.get_encoding('cl100k_base')
		separator_len = len(encoding.encode(SEPARATOR))
		
		relevant_notes = df_search(df, question, total, pprint=True)
		chosen_sections = []
		chosen_sections_len = 0

		for _, row in relevant_notes.iterrows():
				notes_str = f"Path: {row['file_path']}\nSummary:\n{row['summary']}"
				notes_str_len = len(encoding.encode(notes_str))
				if chosen_sections_len + separator_len + notes_str_len > MAX_SECTION_LEN:
						break

				chosen_sections.append(SEPARATOR + notes_str)
				chosen_sections_len += separator_len + notes_str_len

		
		chosen_sections_str = f"".join(chosen_sections)
		print(f"Selected {len(chosen_sections)} document sections:")
		return f'''<|start_context|>\n Project notes to help assistant with answering query "{question}" \n context: {chosen_sections_str}\n<|end_context|>\n<|im_start|>'''


def chatbot(prompt=""):
			EMBEDDING_ENCODING = 'cl100k_base'
			encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
			enc_prompt  = encoder.encode(prompt)
			codebaseContext = q_and_a(question=prompt)
			cbc_prompt  = encoder.encode(codebaseContext)

			print(f"\033[1;37m{enc_prompt}\t\tTokens:{str(len(enc_prompt) + len(cbc_prompt) )}\033[0m")
			avail_tokens= 3596 - (len(enc_prompt)  + len(cbc_prompt))
			print(f"\n\033[1;37mTOTAL OUTPUT TOKENS AVAILABLE:{avail_tokens}\n\033[0m")
			r = requests.post(
					chat_base, headers=headers, stream=True,
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

if __name__ == '__main__':
	chatbot("what is the best way to change this to a stanalone app?")
	q_and_a("API")

	y = df_search(df, "sdk", 1, pprint=True)

	df = pd.read_pickle('summary_pickled.pkl')
	
	df
	print(x, y)

	df['summary_embedding'] = df['summary'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
	

	df["file_path"]=df["file_path"].str.replace("/Downloads/whopt","")
	df_search( df, "security")
	last_result = generate_summary(context_code_pairs,df )
	df["summary"][0]



	def ask(query, gpt=False):
		print(f"{query}\n\nUSER:", flush=False, end="  ")
		question = input()
		if question == "q":
			print("bye")
			exit(0)
			chatbot(prompt=f"{question}")
			print("\n")
			print("Sorry, I didn't understand that.")

	while True:
		print("QUERY:", flush=False, end="  ")
		question = input()
		ask(input)

