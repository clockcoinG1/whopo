import openai
TOKEN_COUNT = 500
root_dir = ""
proj_dir = "llama"
oai_api_key_embedder= "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"
base = "https://api.openai.com/v1/completions"
chat_base = 'https://api.openai.com/v1/chat/completions'
GPT_MODEL = os.getenv("GPT_MODEL") or'gpt-4'
# GPT_MODEL = 'gpt-4-0314'
EMBEDDING_ENCODING = 'cl100k_base'
api_key = "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"
headers = {
	'Content-Type': 'application/json',
	'Authorization': 'Bearer ' + api_key,
}
