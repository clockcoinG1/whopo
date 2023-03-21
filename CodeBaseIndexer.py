import datetime
import os
import pandas as pd
from tqdm import tqdm
from pandas.errors import EmptyDataError
from openai.embeddings_utils import cosine_similarity, get_embedding
import openai 
from constants import oai_api_key_embedder, root_dir, proj_dir

openai.api_key = oai_api_key_embedder

class CodebaseIndexer:
    def __init__(self):
        self.code_root = root_dir + proj_dir
        self.df = None

    def indexCodebase(self, df, pickle="split_codr.pkl"):
        try:
            if not os.path.exists(f"{self.code_root}/{pickle}"):
                df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
                df.to_pickle(f"{self.code_root}/{pickle}")
                self.df = df
                print("Indexed codebase: " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            else:
                self.df = pd.read_pickle(f"{self.code_root}/{pickle}")
        except EmptyDataError as e:
            print(f"Empty data error: {e}")
        except Exception as e:
            print(f"Failed to index codebase: {e}")
        else:
            print("Codebase indexed successfully")