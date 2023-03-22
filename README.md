```# whopo
conda env create -f environment.yml
or
pip install -r requirements.txt

# Code Summarization Chatbot

This is a code summarization chatbot that uses OpenAI's GPT-3 API to summarize code. It is designed to help developers optimize and analyze codebases. 

## Usage

To use the chatbot, run the following command:

```python chatbot.py [-h] [--root ROOT] [-n N] [--prompt PROMPT] [--chat CHAT] [--context CONTEXT] [--max_tokens MAX_TOKENS]
                  directory```

where `directory` is the directory containing the codebase you want to summarize.

The chatbot will then prompt you for a question or query about the codebase. It will then generate a summary of the codebase based on your query.
You can also set the `CODE_EXTRACTOR_DIR` environment variable in the command line by using the `--root` flag:

```python chatbot.py my_project --prompt "What does this code do?" -n 5 --context 10 --max_tokens 500 --root /home/user```

This will set the `CODE_EXTRACTOR_DIR` environment variable to `/home/user`.

You can also set the `GPT_MODEL` environment variable to `gpt-4` or `gpt-4-0314` or `chat-davinci-003-alpha` in the `.env` file in the root directory of the project.

You can use chat mode by adding the `--chat` flag to the command:

```python chatbot.py my_project --prompt "What does this code do?" -n 5 --context 10 --max_tokens 500 --chat```
## Arguments

The chatbot uses the following arguments:

- `directory`: The directory containing the codebase you want to summarize.
- `--root`: Where root of project is or env $CODE_EXTRACTOR_DIR.
- `-n`: The number of context chunks to use.
- `--prompt`: The GPT prompt.
- `--chat`: Whether to use chat mode.
- `--context`: The context length.
- `--max`: The maximum number of in the summary.

You can set these arguments in the command line or in a `.env` file in the root directory of the project.

## Environment Variables

The chatbot uses the following environment variables:

- `CODE_EXTRACTOR_DIR`: The root directory of the codebase you want to summarize.
- `OAI_CD3_KEY`: The OpenAI API key for the Code-Davinci-003 model.
- `OPENAI_API_KEY`: The OpenAI API key for the GPT-3 API.
- `MAX_TOKEN_MAX_SUMMARY`: The maximum number of tokens to use when summarizing code.
- `TOKEN_MAX_SUMMARY`: The number of tokens to use when summarizing code.

You can set these environment variables in a `.env` file in the root directory of the project.

## Requirements

The chatbot requires the following Python packages:

- `numpy`
- `pandas`
- `requests`
- `tqdm`
- `openai`
- `tiktok`
- `pathlib`
- `argparse`

You can install these packages using `pip`:

{```pip install -r requirements.txt```}

This is a chatbot that summarizes code using the OpenAI API. It can be used to summarize code in a directory, or to answer questions about code. 

## License

This project is licensed under the SMD License.




## Authors

- [Twanz](https://github.com/clockcoing1)
- [Kim Jong Un](https://github.com/kjp)
- [Xi Jin Pin](https://github.com/xjp)

## TODO

- Add more features to the chatbot.
- Improve the accuracy of the code summarization.
- Add support for other programming languages.
- Add support for other code summarization models.
- Add support for other chatbot models.

```

USER: Fix the above readme to include the arguments:
`usage: chatbot.py [-h] [--root ROOT] [-n N] [--prompt PROMPT] [--chat CHAT] [--context CONTEXT] [--max_tokens MAX_TOKENS]
                  directory

Code summarization chatbot

positional arguments:
  directory             directory to summarize

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           Where root of project is or env $CODE_EXTRACTOR_DIR
  -n N                  number of context chunks to use
  --prompt PROMPT       gpt prompt
  --chat CHAT           gpt chat
  --context CONTEXT     context length
  --max_tokens MAX_TOKENS
                        maximum number of tokens in summary
(diart) `
and reflect the .env CODE_EXTRACTOR_DIR OAI_CD3_KEY OPENAI_API_KEY MAX_TOKEN_MAX_SUMMARY GPT_MODEL TOKEN_MAX_SUMMARY, and  the command line arguments from the code :
```def main():
		parser = argparse.ArgumentParser(description='Code summarization chatbot')
		parser.add_argument('directory', type=str, default="/ezcoder", help='directory to summarize')
		parser.add_argument('--root', type=str, default=f"{os.environ['CODE_EXTRACTOR_DIR']}", help='Where root of project is or env $CODE_EXTRACTOR_DIR')
		parser.add_argument('-n', type=int, default=10, help='number of context chunks to use')
		parser.add_argument('--prompt', type=str, default='What does this code do?', help='gpt prompt')
		parser.add_argument('--chat', type=bool, default=True, help='gpt chat')
		parser.add_argument('--context', type=int, default=10, help='context length')
		parser.add_argument('--max_tokens', type=int, default=1000, help='maximum number of tokens in summary')

		"""     # ======================= # Help-formatting methods # ======================= def format_usage(self): formatter = self._get_formatter() formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups) return formatter.format_help() def format_help(self): formatter = self._get_formatter() # usage formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups) # description formatter.add_text(self.description) # positionals, optionals and user-defined groups for action_group in self._action_groups: formatter.start_section(action_group.title) formatter.add_text(action_group.description) formatter.add_arguments(action_group._group_actions) formatter.end_section() # epilog formatter.add_text(self.epilog) # determine help from format above return formatter.format_help() def _get_formatter(self): return self.formatter_class(prog=self.prog) # ===================== # Help-printing methods # ===================== def print_usage(self, file=None): if file is None: file = _sys.stdout self._print_message(self.format_usage(), file) def print_help(self, file=None): if file is None: file = _sys.stdout self._print_message(self.format_help(), file) def _print_message(self, message, file=None): if message: if file is None: file = _sys.stderr file.write(message) # =============== # Exiting methods # =============== def exit(self, status=0, message=None): if message: self._print_message(message, _sys.stderr) _sys.exit(status) def error(self, message): error(message: string) Prints a usage message incorporating the message to stderr and exits. If you override this in a subclass, it should not return -- it should either exit or raise an exception. """
		args = parser.parse_args()

		if not os.path.isdir(f'{args.root}/{args.directory}'):
			parser.error(f"The directory specified does not exist.{args.root}/{args.directory}")
		# For argparser lets use its  error handling, exit, help and usage formatting and outputting methods from argparse documentation above. Only output code for the main def argparser code for brevity:		if 
		if not os.path.isdir(args.root):
			parser.error("The root directory specified does not exist.")
		if not os.path.isdir(args.directory):
			parser.error("The directory specified does not exist.")
		if not isinstance(args.n, int):
			parser.error("The number of context chunks must be an integer.")
		if  not isinstance(args.context, int):
			parser.error("The context length must be an integer.")
		if not isinstance(args.max_tokens, int):
			parser.error("The maximum number of tokens must be an integer.")
		if not isinstance(args.prompt, str):
			parser.error("The prompt must be a string.")
		if args.n < 1:
			parser.error("The number of context chunks must be greater than 0.")
		if args.context < 1:
			parser.error("The context length must be greater than 0.")
		if args.max_tokens < 1:
			parser.error("The maximum number of tokens must be greater than 0.")
		if len(args.prompt) < 1:
			parser.error("The prompt must be non-empty.")

		print(f"\033[1;32;40m\nSummarizing {args.directory}")
		print(f"\033[1;32;40m\nUsing {args.n} context chunks")
		print(f"\033[1;32;40m\nPrompt: {args.prompt}")

		proj_dir = args.directory.strip() if args.directory is not None else "ez11"
		root_dir = args.root.strip() if args.root is not None else os.getcwd()
		prompt = args.prompt.strip()  if args.prompt is not None else "Explain the code"
		n = args.n if args.n is not None else 20
		context =  args.context if args.context is not None else 15
		max_tokens = args.max_tokens if args.max_tokens is not None else MAX_TOKEN_MAX_SUMMARY
		if not os.path.exists(root_dir + "/" + proj_dir):
				print(f"Directory {root_dir + args.directory} does not exist")
				sys.exit()
		
		ce = CodeExtractor(f"{root_dir}/{proj_dir}")
		df = ce.get_files_df()
		df = split_code_by_TOKEN_MAX_SUMMARY(df,  col_name="code",  max_tokens=max_tokens) # OR  df = ce.split_code_by_lines(df, max_lines=6)
		df = indexCodebase(df,"code" , pickle=f"{root_dir}/{proj_dir}.pkl", code_root=f"{root_dir}/{proj_dir}")
		print(f"\033[1;32;40m*" * 20 + "\tGenerating summary...\t" + f"\033[1;32;40m*" * 25)
		df = df[df['code'] != '' ].dropna()
		# df.apply(lambda x: print(x["summary"]), axis=1)
		df = generate_summary(df,  proj_dir=proj_dir)
		df = df[df['summary'] != '' ].dropna()
		print(f"\033[1;32;40m*" * 10 + "\tWriting summary...\t" + f"\033[1;32;40m*" * 10)
		write_md_files(df, f"{proj_dir}".strip('/'))
		proj_dir_pikl = re.sub(r'[^a-zA-Z]', '', f"{root_dir}/{proj_dir}")
		
		print(f"\033[1;34;40m*" * 20 + "\t Embedding summary column ...\t" + f"{root_dir}/{proj_dir}"  + f"\033[1;34;40m*" * 20)
		df['summary_embedding'] = df['summary'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002') if x else None)

		print(f"\033[1;32;40m*" * 40 + "\t Saving embedding summary...\t" + f"{root_dir}/{proj_dir}"  + f"\033[1;32;40m*" * 40)
		df.to_pickle(proj_dir_pikl + '.pkl')
		if args.chat: 
			print(f"\033[1;32;40m*" * 10 + "\t Chat mode \t" + f"{root_dir}/{proj_dir}"  + f"\033[1;32;40m*" * 10)
			while True:
				ask = input("\n\033[33mAsk about the files, code, summaries:\033[0m\n\n\033[44mUSER:  \033[0m")
				# q_and_a(df, "What is the code do?", n, 500)# max_tokens * context_n = 15)
				summary_items  = df_search_sum(df, ask, pprint=True, n=n , n_lines=context) 
				chatbot(df, summary_items , context)

if __name__ == '__main__':
	main()

```
ASSISTANT: OK. Here is the fixed README.md