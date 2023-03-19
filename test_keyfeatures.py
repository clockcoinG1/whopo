from pathlib import Path
import openai
import requests
import tiktoken
import pandas as pd
import os
import json
from get_rel_code import api_key

root_dir = '/Users/clockcoin'

def generate_tab2(context_code_pairs, df,model="chat-davinci-003-alpha"):
    message = ""
    try:
        if not model:
            model="chat-davinci-003-alpha"
    except NameError:
        model="chat-davinci-003-alpha"
    
    comp_type = "finish_reason" if not model or model != "chat-davinci-003-alpha" else "finish_details"
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
        print(f"{file_path}")
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
                            print("\n", flush=False, end="")
                            message.strip()
                            continue
        # Add the summary to the DataFrame, matching the file_path
        df.loc[df['file_path'] == filepath, 'summary'] = summary.strip()
    return message

df["summary"][0]


last_result= generate_tab2(context_code_pairs,df )