# Code Summarization Chatbot

This is a chatbot that summarizes code using the OpenAI API. It can be used to summarize code in a directory, or to answer questions about code. 

## Usage

To use the chatbot, run the following command:

```
python chatbot.py <directory>
```

where `<directory>` is the directory containing the code to summarize.

You can also specify a context prompt and a GPT prompt using the `--context` and `--gpt` options, respectively. For example:

```
python chatbot.py <directory> --context "Important code" --gpt "What does this code do?"
```

## Requirements

This script requires the following Python packages:

- `numpy`
- `pandas`
- `requests`
- `tqdm`
- `openai`
- `tiktoken`

## License

This code is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgments

This script was inspired by the [Code Summarization Chatbot](https://github.com/llSourcell/code_summarization_chatbot) by Siraj Raval.

## TODO

- [ ] Add support for other programming languages
- [ ] Improve the quality of the summaries
- [ ] Add support for more complex questions
- [ ] Add support for more complex codebases
- [ ] Add support for more complex GPT prompts
- [ ] Add support for more complex context prompts
- [ ] Add support for more complex chatbot interactions
- [ ] Add support for more complex chatbot responses
- [ ] Add support for more complex chatbot prompts
- [ ] Add support for more complex chatbot contexts
- [ ] Add support for more complex chatbot summaries
- [ ] Add support for more complex chatbot embeddings
- [ ] Add support for more complex chatbot tokenization
- [ ] Add support for more complex chatbot models
- [ ] Add support for more complex chatbot APIs
- [ ] Add support for more complex chatbot data structures
- [ ] Add support for more complex chatbot algorithms
- [ ] Add support for more complex chatbot libraries
- [ ] Add support for more complex chatbot frameworks

```python
# Code Summarization Chatbot

This is a chatbot that summarizes code using the OpenAI API. It can be used to summarize code in a directory, or to answer questions about a codebase.

## Usage

To use the chatbot, run the following command:

```
python chatbot.py <directory>
```

where `<directory>` is the directory containing the code to summarize.

You can also specify a context prompt and a GPT prompt using the `--context` and `--gpt` options, respectively.

## Requirements

This project requires Python 3.6 or later, as well as the following packages:

- `numpy`
- `pandas`
- `requests`
- `tqdm`
- `openai`
- `tiktoken`
```

To install deps
``` conda env create -f environment.yml```
or
``` pip install -r requirements,txt```

To run it:
```shell python test_keyfeatures.py llama --root $PWD ```

