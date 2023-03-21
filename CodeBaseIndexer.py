import datetime
import os
import pandas as pd
from tqdm import tqdm
from pandas.errors import EmptyDataError
from openai.embeddings_utils import cosine_similarity, get_embedding
import openai 
from constants import oai_api_key_embedder, root_dir, proj_dir

openai.api_key = oai_api_key_embedder

def indexCodebase(df, col_name, pickle="split_codr"):
				code_root = root_dir + proj_dir
				try:
					if not os.path.exists(f"{code_root}/{pickle}.pkl"):
						df[f"{col_name}_tokens"] = [list(tokenizer.encode(code)) for code in df[col_name]]
						df[f"{col_name}_token_count"] = [len(code) for code in df[f"{col_name}_tokens"]]
						df[f"{col_name}_embedding"] = df[col_name].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
						df.to_pickle(f"{code_root}/{pickle}.pkl")
						print("Indexed codebase: " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
						return df
					else:
							df = pd.read_pickle(f"{code_root}/{pickle}.pkl")
							return df
				except EmptyDataError as e:
						print(f"Empty data error: {e}")
				except Exception as e:
						print(f"Failed to index codebase: {e}")
				else:
						print("Codebase indexed successfully")