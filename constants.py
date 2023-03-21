import os
TOKEN_COUNT = 500
root_dir = os.path.expanduser("~")
proj_dir = "llama"
oai_api_key_embedder= "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"
base = "https://api.openai.com/v1/completions"
chat_base = 'https://api.openai.com/v1/chat/completions'
EMBEDDING_ENCODING = 'cl100k_base'
api_key = "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"
headers = {
	'Content-Type': 'application/json',
	'Authorization': 'Bearer ' + api_key,
}
