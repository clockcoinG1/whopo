import os
from dotenv import load_dotenv

load_dotenv()

MAX_TOKEN_MAX_SUMMARY = os.getenv("MAX_TOKEN_MAX_SUMMARY") or 3000
TOKEN_MAX_SUMMARY = os.getenv("TOKEN_MAX_SUMMARY") or 3000
root_dir = os.getenv("CODE_EXTRACTOR_DIR") or os.getcwd()
proj_dir = "ez11"
oai_api_key_embedder = os.getenv("OPENAI_API_KEY")
base = "https://api.openai.com/v1/completions"
chat_base = 'https://api.openai.com/v1/chat/completions'
EMBEDDING_ENCODING = 'cl100k_base'
api_key = os.getenv("OPENAI_API_KEY")
comp_api_key = os.getenv("OAI_CD3_KEY") or "sk-"
GPT_MODEL = os.getenv("GPT_MODEL") or "gpt-4"
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + comp_api_key,
}
