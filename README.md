# Code Summarization Chatbot

This chatbot helps you understand and navigate through a codebase by generating summaries and answering questions about the code. It uses GPT-3 to generate summaries and answer questions based on the code context.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Command-line Arguments](#command-line-arguments)
- [Environment Variables](#environment-variables)
- [Requirements](#requirements)
- [License](#license)
- [Authors](#authors)
- [TODO](#todo)

## Prerequisites

- Python 3.6 or higher
- Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-repo/code-summarization-chatbot.git
cd code-summarization-chatbot
pip install -r requirements.txt
```

## Usage

1. Navigate to the project directory.

2. Run the `app.py` script with the appropriate command-line arguments. Here's an example:

```bash
python app.py --directory /path/to/codebase --ext py,ts,js,md,txt --n 10 --context 10 --max_tokens 1000
```

After running the script, you can interact with the chatbot by typing your questions in the terminal.

### Command-line Arguments

- `--directory`: The directory containing the codebase you want to summarize (default: current working directory).
- `-P`: The path to a saved database file (pickle) to use for the chatbot.
- `--root`: The root directory of the project (default: environment variable `$CODE_EXTRACTOR_DIR`).
- `-n`: The number of context chunks to use (default: 10).
- `--prompt`: The GPT-3 prompt (default: "What does this code do?").
- `--chat`: Enable GPT-3 chat mode (default: True).
- `--context`: The context length (default: 10).
- `--max_tokens`: The maximum number of tokens in the summary (default: 1000).
- `--ext`: The file extensions to target (default: "py,ts,js,md,txt").
- `--split_by`: Choose to split code by tokens or lines (default: "lines").

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

```bash
pip install -r requirements.txt
```

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
- Add support for other code summaries