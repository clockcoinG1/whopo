import numpy as np
from pathlib import Path
import requests
import tiktoken
import pandas as pd
import os
import re
import json
from get_rel_code import api_key, get_context_code, get_rel_context_summary
import tqdm
from CodebaseIndexer import indexCodebase
from embedder import CodeExtractor
import openai
from openai.embeddings_utils import cosine_similarity, get_embedding
import uuid
from constants import (
	TOKEN_COUNT,
	root_dir,
	MAX_TOKEN_COUNT,
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

def generate_summary(df,model="chat-davinci-003-alpha", proj_dir = "llama"):
	df["file_path"]=df["file_path"].str.replace(os.getenv("HOME") + root_dir + proj_dir,"")
	message = ""
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
		prompt = f"\nSYSTEM: You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up.\nUSER:{code}\nUSER:Please summarize the key features of the specified file within the project directory, and present the information in a concise bullet-point format. Focus on aspects such as the file's content.\nASSISTANT: Sure, here are the key features of the {filepath} file:\n - "
		encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
		enc_prompt = encoder.encode(str(prompt))
		tokens = len(encoder.encode(code) + enc_prompt) or 1
		max_token = 500 + MAX_TOKEN_COUNT
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
				# "logit_bias": {
				# 	"[27, 91, 320, 62, 437, 91, 29, 198]" : -100
				# 	},
				"stop": ["\nSYSTEM:", "\nUSER:", "\nASSISTANT:","<|im_end|>" ],
				"max_tokens": 3000 + tokens,
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
				df.loc[df['file_name'] == filename, 'summary'] = summary.strip()
			except KeyError:
				df['summary'] = ''
				df.loc[df['file_name'] == filename, 'summary'] = summary.strip()

	df  = df[pd.notna(df['summary'])]
	proj_dir_pikl = re.sub(r'[^a-zA-Z]', '', proj_dir)
	try: 
		df["summary_tokens"] = [list(tokenizer.encode(summary)) for summary in df["summary"]]
		df["summary_token_count"] = [len(summary) for summary in df["summary_tokens"]]
		print("Embedding summaries...")
		df['summary_embedding'] = df['summary'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
		df.to_pickle(f"{proj_dir_pikl}.pkl")
		print(f'Saved vectors to "{proj_dir_pikl}.pkl"')
	except:
			print('error getting embedding...')
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
		return f'''<|start_context|>\n Project notes to help assistant with answering query "{question}" \n context: {chosen_sections_str}\n<|end_context|>\n<|im_start|>'''


def chatbot(df, prompt="", n = 4):
			encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
			enc_prompt  = encoder.encode(prompt)
			codebaseContext = q_and_a(df, question=prompt, total = n)
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

def generate_summary_for_directory(directory, df):
    result = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.endswith('.py') or entry.name.endswith('.cpp') or entry.name.endswith('.ts') or entry.name.endswith('.js') or entry.name.endswith('.ant'):
                file_path = os.path.join(directory, entry.name)
                if df[df['file_path'] == file_path]['summary'].empty:
                    summary = generate_summary_for_file(file_path)
                    df.loc[df['file_path'] == file_path, 'summary'] = summary
                    result[file_path] = summary
                else:
                    result[file_path] = df[df['file_path'] == file_path]['summary'].values[0]
    return result



if __name__ == '__main__':
	ce  = CodeExtractor(f"{root_dir}{proj_dir}")
	df = ce.get_files_df()
	df = indexCodebase(df, "code", pickle="test_emb")
	df = generate_summary(df)
	chatbot(df , "models used and todo items",20)
	# df = ce.split_code_by_token_count(df, MAX_TOKEN_COUNT)
	# ce.indexCodebase(df)
	# df = indexCodebase(df, "summary", pickle="test_sum")
