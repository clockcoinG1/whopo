def calculate_cosine_similarity(self, code: str, snippets: List[str]) -> pd.DataFrame:
				"""
				Calculate cosine similarity between code snippet and all snippets in the database
				"""
				similarities = [cosine_similarity(code, snippet) for snippet in snippets]
				return pd.DataFrame({"text": snippets, "similarity": similarities})

		def generate_completions(self, prompt: str, max_tokens: int = 1024) -> str:
				"""
				Generate text completions from prompt
				"""
				data = {
						"prompt": prompt,
						"temperature": 0.3,
						"max_tokens": max_tokens,
						"stop": "<|im_sep|>",
						"model": self.COMPLETIONS_MODEL,
						"n": 1,
						"stream": False,
						"logprobs": None,
				}
				json_response = requests.post(url=self.base, headers=self.headers, data=json.dumps(data))
				response_text = json.loads(json_response.text)["choices"][0]["text"]
				return response_text.strip()

		def generate_function_docstring(self, function_name: str, function_code: str) -> str:
				"""
				Generate function docstring
				"""
				prompt = f'"""{self.SEPARATOR} Docstring for {function_name} {self.SEPARATOR}{function_code}{self.SEPARATOR}"""'
				return self.generate_completions(prompt)

		def generate_class_docstring(self, class_name: str, class_code: str) -> str:
				"""
				Generate class docstring
				"""
				prompt = f'"""{self.SEPARATOR} Docstring for {class_name} {self.SEPARATOR}{class_code}{self.SEPARATOR}"""'
				return self.generate_completions(prompt)

		def extract_snippets_from_df(self, df: pd.DataFrame) -> List[str]:
				"""
				Extract code snippets from DataFrame
				"""
					return list(df["code"])

		def generate_docstrings(self, df: pd.DataFrame) -> pd.DataFrame:
				"""
				Generate docstrings for each function and class in DataFrame
				"""
				docstrings = []
				snippets = self.extract_snippets_from_df(df)

				for index, row in df.iterrows():
						is_function = row["function_name"] is not None and row["function_name"] != ""
						is_class = row["class_name"] is not None and row["class_name"] != ""

						if is_function:
								function_name = row["function_name"]
								function_code = row["code"]
								docstring = self.generate_function_docstring(function_name, function_code)
								docstrings.append({"name": function_name, "docstring": docstring.strip()})
						elif is_class:
								class_name = row["class_name"]
								class_code = row["code"]
								docstring = self.generate_class_docstring(class_name, class_code)
								docstrings.append({"name": class_name, "docstring": docstring.strip()})

						if index > 0 and index % 100 == 0:
								print(f"Completed {index} of {len(df)}")

				docstrings_df = pd.DataFrame(docstrings)
				return docstrings_df

		def get_docstrings(self, df: pd.DataFrame) -> pd.DataFrame:
				docstrings = []
				for index, row in df.iterrows():
						is_function = row["function_name"] is not None and row["function_name"] != ""
						is_class = row["class_name"] is not None and row["class_name"] != ""

						if is_function:
								function_name = row["function_name"]
								docstring = self.get_function_docstring(function_name)
								docstrings.append({"name": function_name, "docstring": docstring.strip()})
						elif is_class:
								class_name = row["class_name"]
								docstring = self.get_class_docstring(class_name)
								docstrings.append({"name": class_name, "docstring": docstring.strip()})

						if index > 0 and index % 100 == 0:
								print(f"Completed {index} of {len(df)}")

				docstrings_df = pd.DataFrame(docstrings)
				return docstrings_df

		def get_function_docstring(self, function_name: str) -> str:
				"""
				Get docstring for function
				"""
				result = self.generate_completions(f"function {function_name}", max_tokens=1024)
				return result

		def get_class_docstring(self, class_name: str) -> str:
				"""
				Get docstring for class
				"""
				result = self.generate_completions(f"class {class_name}", max_tokens=1024)
				return result

		def cosine_similarity(self, embeddings: List, prompt: str, k: int = 10) -> pd.DataFrame:
				"""
				Calculate cosine similarity between embeddings and prompt
				"""
				similarity_scores = [cosine_similarity(embedding, prompt) for embedding in embeddings]
				df = pd.DataFrame({"similarity": similarity_scores})
				df_sorted = df.sort_values("similarity", ascending=False).reset_index(drop=True)
				return df_sorted.head(k)

		def search_functions(self, query: str, k: int = 10) -> pd.DataFrame:
				"""
				Search functions based on query
				"""
				results = pd.DataFrame(columns=["function_name", "code", "similarity"])
				functions = self.df["function_name"]
				embeddings = self.df["code_embedding"]
				query_embedding = get_embedding(query, engine=self.EMBEDDING_MODEL)

				if query_embedding is not None:
						similarity_scores = [cosine_similarity(embedding, query_embedding) for embedding in embeddings]
						df = pd.DataFrame({"function_name": functions, "code": self.df["code"], "similarity": similarity_scores})
						df_sorted = df.sort_values("similarity", ascending=False).reset_index(drop=True)
						results = df_sorted.head(k)

				return results

		def search_docstrings(self, query: str, k: int = 10) -> pd.DataFrame:
				"""
				Search docstrings based on query
				"""
				results = pd.DataFrame(columns=["function_name", "docstring", "similarity"])
				docstrings = self.df_docstrings["docstring"]
				names = self.df_docstrings["name"]
				embeddings = [get_embedding(docstring, engine=self.EMBEDDING_MODEL) for docstring in docstrings]
				query_embedding = get_embedding(query, engine=self.EMBEDDING_MODEL)

				if query_embedding is not None:
						df_similarity = self.cosine_similarity(embeddings, query_embedding)
						df_names = pd.DataFrame({"function_name": names})
						df = pd.concat([df_names, df_similarity], axis=1)
						results = df.sort_values("similarity", ascending=False).reset_index(drop=True).head(k)

				return results


def get_code_snippets(lines, line_range):
		start = line_range[0]
		end = line_range[1]
		code_snippets = " ".join(lines[start:end + 1])
		return code_snippets


def get_similar_codes(code, snippets, prompt, k):
		similarities = cosine_similarity(code, snippets)
		sim_df = pd.DataFrame({"text": snippets, "similarity": similarities})
		top_k_sim_df = sim_df.sort_values("similarity", ascending=False).head(k)
		top_k_sim_df["prompt"] = prompt
		top_k_sim_df = top_k_sim_df[["prompt", "text", "similarity"]]
		return top_k_sim_df


def search_code(df, query, k):
		snippets = df["code"].tolist()
		similarities = cosine_similarity(query, snippets)
		sim_df = pd.DataFrame({"code": snippets, "similarity": similarities})
		top_k_sim_df = sim_df.sort_values("similarity", ascending=False).head(k)
		return top_k_sim_df


def to_altair(df):
		df_chart = df.sort_values(by="similarity", ascending=False)
		chart = (
				alt.Chart(df_chart)
				.mark_bar()
				.encode(x="text", y="similarity", tooltip="text")
				.configure_axisX(labelAngle=-45, labelAlign="left")
				.properties(width=800, height=300)
		)
		return chart


if __name__ == "__main__":
		try:
				extractor = CodeExtractor(code_root)
				# df = extractor.get_all_functions()
				# df.to_pickle("df.pkl")
				# df = pd.read_pickle("df.pkl")
				# df = extractor.split_code_by_token_count(df, max_tokens=4000)
				# df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
				# df.to_pickle("df2.pkl")
				df = pd.read_pickle(f"{root_dir}/df3.pkl")
				docstrings_df = extractor.generate_docstrings(df)
				docstrings_df.to_pickle("docstrings.pkl")
				docstrings_df.to_csv("docstrings.csv", index=False)
				print("Finished generating docstrings")
				# df = pd.read_pickle("df.pkl")
				snippets = df["code"].tolist()
				prompt = '""" Docstring'
				top_k_sim_df = get_similar_codes(snippets, prompt, 10)
				print(top_k_sim_df)

		except Exception as ex:
				print(ex)
				traceback.print_exc()
				sys.exit(1)
		sys.exit(0)