
USER: The following method is returning this error:

```
The following method is giving this error:
```
>>> extractor.split_code_by_token_count(extractor.df, 100)
Token count: 129
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 270, in split_code_by_token_count
AttributeError: 'int' object has no attribute 'value'
```


```
class CodeExtractor:
... ommited for brevity... 
def split_code_by_token_count(self, df: pd.DataFrame, max_tokens: int = 8100) -> pd.DataFrame:
		"""Use the same tokenizer as the pre-trained model"""
		EMBEDDING_ENCODING = 'cl100k_base'
		tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)
		new_rows = []

		# Iterate over all rows in the dataframe
		for index, row in df.iterrows():
				code = row["code"]
				# Tokenize the code
				tokens = list(tokenizer.encode(code))
				# Get the number of tokens in the code
				token_count = len(tokens)
				print(token_count)

				if token_count <= max_tokens:
						new_rows.append(row)
				else:
						# Get the halfway point of the code
						halfway_point = token_count // 2
						# Find the first occurrence of two newline characters in the code
						split_point = None

						# We have added the verbose debugging to see the value of some variables while running the code.
						# In each iteration, we print the token count, i.e., how many tokens we have in the code.
						# If the token count is greater than max_tokens, we will try to split the code into two parts.
						# The goal is to create two sets of codes with as few a number of tokens as possible but not exceeding the max_tokens limit.
						print(f"Token count: {token_count}")
						for i in range(halfway_point, len(tokens) - 1):
								if tokens[i].value == "\n" and tokens[i + 1].value == "\n":
										split_point = i
										break
						# If we found a place to split the code, then we create two new rows with the first set of codes and the second set of codes, respectively.
						if split_point:
								# Split the code into two parts
								first_half = "".join(token.value for token in tokens[: split_point + 1])
								second_half = "".join(token.value for token in tokens[split_point + 1 :])

								# Create a new row for the first half
								new_row1 = row.copy()
								new_row1["code"] = first_half
								new_rows.append(new_row1)

								# Create a new row for the second half
								new_row2 = row.copy()
								new_row2["code"] = second_half
								new_rows.append(new_row2)
						# If we are unable to split the code, then we keep the code as it is.
						else:
								new_rows.append(row)

		return pd.DataFrame(new_rows).reset_index(drop=True)
```
USER:Split all the code into 8090 character chunks and reindex the  Below is the datame
```
         file_name                                          file_path                                               code                                             tokens  token_count
663   extension.js  /Users/clockcoin/codex-vscode/ezcoder/dist/ext...                                                 },                                             [2186]            1
1596  extension.js  /Users/clockcoin/codex-vscode/ezcoder/dist/ext...                                             events                                            [12670]            1
1597  extension.js  /Users/clockcoin/codex-vscode/ezcoder/dist/ext...                                                 fs                                             [3933]            1
1608  extension.js  /Users/clockcoin/codex-vscode/ezcoder/dist/ext...                                               util                                             [1338]            1
180   extension.js  /Users/clockcoin/codex-vscode/ezcoder/dist/ext...                                                 },                                             [2186]            1
...            ...                                                ...                                                ...                                                ...          ...
1185  extension.js  /Users/clockcoin/codex-vscode/ezcoder/dist/ext...  e.subscriptions.push(r.commands.registerComman...  [68, 86247, 2615, 2666, 35002, 10131, 4153, 44...          479
1130  extension.js  /Users/clockcoin/codex-vscode/ezcoder/dist/ext...  return{cancelFineTune:(e,a)=>t.cancelFineTune(...  [693, 90, 19022, 64816, 51, 2957, 3349, 68, 15...          551
1131  extension.js  /Users/clockcoin/codex-vscode/ezcoder/dist/ext...  class r extends o.BaseAPI{cancelFineTune(e,i){...  [1058, 436, 2289, 297, 13316, 7227, 90, 19022,...          928
1518  extension.js  /Users/clockcoin/codex-vscode/ezcoder/dist/ext...  \t"}" >\n\t\t\t\t\t\t < div class="tooltip" > ...  [197, 9388, 1, 871, 59, 77, 5061, 5061, 5061, ...         1037
1577  extension.js  /Users/clockcoin/codex-vscode/ezcoder/dist/ext...  \te.exports = JSON.parse( '{"application/1d-in...  [7883, 7587, 284, 4823, 4736, 7, 53198, 5242, ...        35676
````
ASSISTANT:
  