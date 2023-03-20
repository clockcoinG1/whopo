import numpy as np
from pathlib import Path
<<<<<<< HEAD
import openai
=======
>>>>>>> cc87d40 (added configuration.py and cleaning up excess code)
import requests
import tiktoken
import pandas as pd
import os
import json
<<<<<<< HEAD
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

=======
from get_rel_code import api_key, get_context_code
import tqdm
from embedder import CodeExtractor
from openai.embeddings_utils import cosine_similarity, get_embedding
import uuid
from constants import  TOKEN_COUNT, root_dir, proj_dir, oai_api_key_embedder, base, chat_base, EMBEDDING_ENCODING, headers
openai.api_key = oai_api_key_embedder
>>>>>>> cc87d40 (added configuration.py and cleaning up excess code)

def generate_summary(context_code_pairs, df,model="chat-davinci-003-alpha"):
	message = ""
	try:
		if not model:
			model="chat-davinci-003-alpha"
	except NameError:
		model="chat-davinci-003-alpha"	
	comp_type = "finish_reason" if not model or model != "chat-davinci-003-alpha" else "finish_details"
	for filepath, code in context_code_pairs:
		# filepath  = filepath.replace("/Downloads/whopt","")
		prompt = f"\nSYSTEM: You are the ASSISTANT helping the USER with optimizing and analyzing a codebase. You are intelligent, helpful, and an expert developer, who always gives the correct answer and only does what is instructed. You always answer truthfully and don't make things up.\nUSER:{code}\nUSER:Please summarize the key features of the specified file within the project directory, and present the information in a concise bullet-point format. Focus on aspects such as the file's content.\nASSISTANT: Sure, here are the key features of the {filepath} file:\n - "
<<<<<<< HEAD
		EMBEDDING_ENCODING = 'cl100k_base'
=======
>>>>>>> cc87d40 (added configuration.py and cleaning up excess code)
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
				# "logit_bias": {
				# 	"[27, 91, 320, 62, 437, 91, 29, 198]" : -100
				# 	},
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
		
<<<<<<< HEAD
		df.loc[df['file_path'] == filepath, 'summary'] = summary.strip()
	return message




=======
		# df.loc[df['file_path'] == filepath, 'summary'] = summary.strip()
	try: 
		with tqdm.pandas() as tq:
			if df[df['file_path'] == file_path]['summary'].empty:
				df.loc[df['file_path'] == filepath, 'summary'] = "NA"
				df.loc[df['file_path'] == filepath, 'summary'] = summary.strip()
				df  = df[pd.notna(df['summary'])]
				print("embedding summaries...\n\n")
				ce.df['summary_embedding'] = ce.df['summary'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
				return message
	except:
		print("embedding error")
		return message
>>>>>>> cc87d40 (added configuration.py and cleaning up excess code)


def get_tokens(df, colname):
	EMBEDDING_ENCODING = "cl100k_base"
	encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
	colname = "summary"
	code_type = list(df[df.columns[df.columns.to_series().str.contains(colname)]])
	for _, row in tqdm.tqdm(df.iterrows()):
		
		filepath = row["file_path"]
<<<<<<< HEAD
		emb_data = "file path: " + filepath + "\n" + row["summary"]
=======
		emb_data = "file path: " + filepath + "\n" + str(row["summary"])
>>>>>>> cc87d40 (added configuration.py and cleaning up excess code)
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




<<<<<<< HEAD
def q_and_a(question = "What isthe most important file", total = 10, MAX_SECTION_LEN = 7000) -> str:
		SEPARATOR = "<|im_sep|>"
		encoding = tiktoken.get_encoding('cl100k_base')
		separator_len = len(encoding.encode(SEPARATOR))
		
		relevant_notes = df_search(df, question, total, pprint=True)
		chosen_sections = []
		chosen_sections_len = 0

=======
def q_and_a(encoding, df, question = "What isthe most important file", total = 10, MAX_SECTION_LEN = 7000) -> str:
		SEPARATOR = "<|im_sep|>"
		separator_len = len(encoding.encode(SEPARATOR))
		relevant_notes = df_search(df, question, total, pprint=True)
		chosen_sections = []
		chosen_sections_len = 0
>>>>>>> cc87d40 (added configuration.py and cleaning up excess code)
		for _, row in relevant_notes.iterrows():
				notes_str = f"Path: {row['file_path']}\nSummary:\n{row['summary']}"
				notes_str_len = len(encoding.encode(notes_str))
				if chosen_sections_len + separator_len + notes_str_len > MAX_SECTION_LEN:
						break
<<<<<<< HEAD

				chosen_sections.append(SEPARATOR + notes_str)
				chosen_sections_len += separator_len + notes_str_len

=======
				chosen_sections.append(SEPARATOR + notes_str)
				chosen_sections_len += separator_len + notes_str_len
>>>>>>> cc87d40 (added configuration.py and cleaning up excess code)
		
		chosen_sections_str = f"".join(chosen_sections)
		print(f"Selected {len(chosen_sections)} document sections:")
		return f'''<|start_context|>\n Project notes to help assistant with answering query "{question}" \n context: {chosen_sections_str}\n<|end_context|>\n<|im_start|>'''


<<<<<<< HEAD
def chatbot(prompt=""):
			EMBEDDING_ENCODING = 'cl100k_base'
			encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
			enc_prompt  = encoder.encode(prompt)
			codebaseContext = q_and_a(question=prompt)
			cbc_prompt  = encoder.encode(codebaseContext)

=======
def chatbot(df, prompt=""):
			encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
			enc_prompt  = encoder.encode(prompt)
			codebaseContext = q_and_a(encoder, df, question=prompt)
			cbc_prompt  = encoder.encode(codebaseContext)
>>>>>>> cc87d40 (added configuration.py and cleaning up excess code)
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
<<<<<<< HEAD
			return message.strip()

if __name__ == '__main__':


	df = pd.read_pickle('summary_pickled.pkl')
	generate_summary(context_code_pairs, df)
	# context_pairs = df_search(df, "sdk", 1, pprint=True)
	# last_result = generate_summary(context_code_pairs,df )
	df['summary_embedding'] = df['summary'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
	df["file_path"]=df["file_path"].str.replace("/Downloads/whopt","")
	df_search( df, "security")
	df["summary"][0]
	chatbot("what is the best way to change this to a stanalone app?")
	q_and_a("API")
=======
				return message.strip()

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

if __name__ == '__main__':
	"""
		Split by token count
		or :
		 df = ce.split_code_by_lines(df,5)
	"""
	ce  = CodeExtractor(f"{root_dir}{proj_dir}")
	ce.df = ce.get_files_df()
	encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
	ce.df = ce.split_code_by_token_count(df, 200)
	ce.indexCodebase(ce.df, pickle=name)
	ce.df["token_count"].sum()

	name = f"codebase_pickle-{str(uuid.uuid4()).split('-')[0]}.pkl"
	ce.df_search(ce.df, "sdk",15, pprint = True) 
	
	context_code_pairs = get_rel_context_summary(root_dir,ce.df , 'import')
	
	# make summary of code 
	generate_summary(context_code_pairs,|)

	# SEARCH for matching summary
	context_pairs = df_search(ce.df, "server", 10, pprint=True)
	ce.df  = get_tokens(ce.df,"summary")
	ce.df.sort_values(["summary"])
	# last_result = generate_summary(context_code_pairs,df )
	df["file_path"]=df["file_path"].str.replace(os.getenv("HOME"),"")
	for file_path, summary in new_summaries.items():
		if summary:
				df.loc[df['file_path'] == filepath, 'summary'] = summary.strip()
# USER: df where "summary" column rows are not NaN
# Assistant: OK here is how to get pandas to replace NaN with an empty string:


	df["summary"][0]
	chatbot("what is the best way to change this to a stanalone app?")
	q_and_a(ce.df, "API")
>>>>>>> cc87d40 (added configuration.py and cleaning up excess code)



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

