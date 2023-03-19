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




last_result= generate_tab2(context_code_pairs,df )

""" - Take the codebase provided.
- Load the text into a DataFrame.
- Pick some code snippets from files in that directory.
- Call the function `get_context_code` which we had defined earlier, in order to extract the context info for the code snippet.
- Take the context information along with some prompt information as input into the function `generate_table`.
- The function `generate_table` should update the DataFrame such that the new column `key_features` for the modified table should have the bullet points describing the file.
 """

# define the root dir


# define the text of all files
text = extract_all_text_files(root_dir)

# create a DataFrame object
df = pd.DataFrame(columns=["file_name", "file_path", "code", "key_features"])

# load all messages into the DataFrame object
context = pd.DataFrame()
for i, (filep, filen, code) in enumerate(get_code_snippets(text)):
    df.loc[i] = {"file_name": filen, "file_path": filep, "code": code.strip(), "key_features": ""}

# extract context from code snippets
context_code_pairs = get_context_code(root_dir, df, "os.path.join")
            
# generate a summary of the key features and update the DataFrame
df = generate_table(context_code_pairs, df)
df.head()

import math

def split_text(text, num_segments):
    segment_size = math.ceil(len(text) / num_segments)
    segments = [text[i:i+segment_size] for i in range(0, len(text), segment_size)]
    return segments

def generate_summary(model="text-davinci-002", top_p=0.5, prompt=None, segments=[], api_key='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'):
    # encode the prompt
    tokenizer = openai.api_key = api_key# your api key here
    encoded_prompt = tokenizer.encode(prompt)

    # define the request headers and data
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}
    data = {"model": model,
            "prompt": encoded_prompt,
            "temperature": 0.7,
            "top_p": top_p,
            "n": 1,
            "stream": False,
            "max_tokens": 500,
            "stop": "\n\n"}

    # generate a summary for each segment
    summaries = []
    for segment in segments:
        # generate a summary for the segment
        data["prompt"] = encoded_prompt + tokenizer.encode(segment.strip())
        response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=data)
        response_data = response.json()

        # extract the summary from the response data
        summary = response_data["choices"][0]["text"].strip()
        summaries.append(summary)

    # join all summaries together
    summarized_summary = "\n".join(summaries)

    return summarized_summary


# define the prompt
prompt = "Please generate a summary of the codebase."

# split the input text into three equal parts
segments = split_text(text, 3)

# generate the summarized summary
summarized_summary = generate_summary(model="text-davinci-002", prompt=prompt, segments=segments, api_key=api_key)

# print the summarized summary
print(summarized_summary)
