# whopo
pip install -r requirements.txt

# Code Summarization Chatbot

This is a code summarization chatbot that uses OpenAI's GPT-3 API to summarize code. It is designed to help developers optimize and analyze codebases. 

## Usage

To use the chatbot, run the following command:

```python chatbot.py [-h] [--root ROOT] [-n N] [--prompt PROMPT] [--chat CHAT] [--context CONTEXT] [--max_tokens MAX_TOKENS] directory```

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
