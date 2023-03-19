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
        base = "https://api.openai.com/v1/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + api_key,
        }
        max_token = 500 + tokens
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
        # Add the summary to the DataFrame, matching the file_path
        df.loc[df['file_path'] == filepath, 'summary'] = summary.strip()
    return message


def get_tokens(df, colname):
    EMBEDDING_ENCODING = "cl100k_base"
    encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
    colname = "summary"
    code_type = list(df[df.columns[df.columns.to_series().str.contains(colname)]])
    for _, row in tqdm.tqdm(df.iterrows()):
        print(row["file_path"] + "\n" + row["summary"], end="\n")
        filepath = row["file_path"]
        emb_data = "file path: " + filepath + "\n" + row["summary"]
        tokens = len(encoder.encode(emb_data )) 
        df.loc[df['file_path'] == filepath, 'tokens_summary'] = tokens
    df[['tokens_summary']] = df[['tokens_summary']].applymap(np.int64)
    return df

def df_search(df, summary_query, n=3, pprint=True, n_lines=7):
    embedding = get_embedding(engine="text-embedding-ada-002", text=summary_query)
    df['summary_similarities'] = df.summary_embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('summary_similarities', ascending=False).head(n)
    res_str = ""
    if pprint:
        for r in res.iterrows():
            print(f"{r[1].file_path}/n {r[1].summary} /n score={r[1].summary_similarities}")
            res_str += f"{r[1].file_path}/n {r[1].summary} /n score={r[1].summary_similarities}"
            res_str += "\n".join(r[1].summary.split("\n")
            print('-' * 70)
    return res

df2 = get_tokens(df, "summary")
df2.sort_values('tokens_summary')
df2['summary_embedding'] = df['summary'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df2.to_pickle("summary_pickled.pkl")
df_search(df2, "server side rendering", 20, pprint=True, n_lines=20)


df = pd.read_pickle('df1.pkl')
#for column fiklepath remove the root dir from each row. "/Downloads/whopt
df["file_path"]=df["file_path"].str.replace("/Downloads/whopt","")

last_result = generate_summary(context_code_pairs,df )
df["summary"][0]
