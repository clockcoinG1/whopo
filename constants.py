MAX_TOKEN_COUNT = 3500
TOKEN_COUNT = 500
root_dir = "/Users/clockcoin"
proj_dir = "/parsero/ezcoder"
oai_api_key_embedder= "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"
base = "https://api.openai.com/v1/completions"
chat_base = 'https://api.openai.com/v1/chat/completions'
EMBEDDING_ENCODING = 'cl100k_base'
api_key = "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"
# comp_api_key = "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"
comp_api_key = "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"
headers = {
	'Content-Type': 'application/json',
	'Authorization': 'Bearer ' + comp_api_key,
}
