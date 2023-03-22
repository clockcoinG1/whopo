
# Code Summarization Chatbot

This is a code summarization chatbot that uses OpenAI's GPT-3 API to summarize code. It is designed to help developers optimize and analyze codebases. 

## Usage

To use the chatbot, run the following command:

```python chatbot.py <directory>```

where `<directory>` is the directory containing the codebase you want to summarize.

The chatbot will then prompt you for a question or query about the codebase. It will then generate a summary of the codebase based on your query.

## Environment Variables

The chatbot uses the following environment variables:

- `CODE_EXTRACTOR_DIR`: The root directory of the codebase you want to summarize.
- `OAI_CD3_KEY`: The OpenAI API key for the Code-Davinci-003 model.
- `OPENAI_API_KEY`: The OpenAI API key for the GPT-3 API.
- `MAX_TOKEN_COUNT`: The maximum number of tokens to use when summarizing code.
- `TOKEN_COUNT`: The number of tokens to use when summarizing code.

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

```pip install -r requirements.txt```

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