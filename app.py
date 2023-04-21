import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding

from chatbot import chatbot, df_search_sum, generate_summary
from constants import EMBEDDING_ENCODING
from glob_files import glob_files
from utils import (indexCodebase, setup_logger, split_code_by_lines,
                   split_code_by_tokens)

tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)


class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def process_arguments():
    parser = argparse.ArgumentParser(description='Code summarization chatbot')
    parser.add_argument('--directory', type=str, default=os.getcwd(), help='directory to summarize')
    parser.add_argument('-P', type=str, default="", help='saved db to use')
    parser.add_argument(
        '--root',
        type=str,
        default=f"{os.environ['CODE_EXTRACTOR_DIR']}",
        help='Where root of project is or env $CODE_EXTRACTOR_DIR',
    )
    parser.add_argument('-n', type=int, default=10, help='number of context chunks to use')
    parser.add_argument('--prompt', type=str, default='What does this code do?', help='gpt prompt')
    parser.add_argument('--chat', type=bool, default=True, help='gpt chat')
    parser.add_argument('--context', type=int, default=10, help='context length')
    parser.add_argument('--max_tokens', type=int, default=1000, help='maximum number of tokens in summary')
    parser.add_argument('--ext', type=str, default="py,ts,js,md,txt", help='file ext to target')
    parser.add_argument(
        '--split_by', type=str, choices="tokens,lines", default='lines', help='split code by tokens or lines'
    )

    return parser.parse_args()


def validate_arguments(args):
    if args.P:
        if not os.path.isfile(args.P):
            raise ValueError(f"The file specified does not exist: {args.P}")

    if not os.path.exists(args.root + '/' + args.directory.strip()):
        raise ValueError(f"Directory {args.root  + '/' + args.directory.strip()} does not exist")


def main():
    logger = setup_logger("SOTA_Logger")
    try:
        args = process_arguments()
        validate_arguments(args)
        proj_dir = Path(args.root.strip(), args.directory.strip())
        Path(args.root.strip())
        args.prompt.strip()
        ext = args.ext
        n = args.n
        context = args.context
        max_tokens = args.max_tokens
        split_by = args.split_by
        if args.P:
            df = pd.read_pickle(args.P)

            while True:
                ask = input(f"\n{TerminalColors.OKCYAN}USER:{TerminalColors.ENDC} ")
                result = df_search_sum(df, ask)
                chatbot(
                    df,
                    (
                        f"{TerminalColors.OKGREEN}{result}{TerminalColors.ENDC}\n\n{TerminalColors.OKCYAN}USER:"
                        f" {ask}{TerminalColors.ENDC}"
                    ),
                )

        else:
            logger.info(f"Summarizing {args.directory}\nUsing {args.n} context chunks\nPrompt: {args.prompt}")
            df = glob_files(str(proj_dir), ext)
            if split_by == 'lines':
                df = split_code_by_lines(df, max_lines=context)
            else:
                df = split_code_by_tokens(df, max_tokens=max_tokens)
            df = df[df['code'] != ''].dropna()
            df = indexCodebase(df, "code")
            logger.info("Generating summary...")
            logger.info("Writing summary...")
            df = generate_summary(df)
            logger.info(f"Saving embedding summary to {proj_dir}")
            proj_dir_pikl = re.sub(r'[^a-zA-Z0-9]', '', f"{proj_dir}.pkl")
            df = df[df['summary'] != ''].dropna()
            logger.info('Processing summaries')
            df['summary_embedding'] = df['summary'].apply(
                lambda x: get_embedding(x, engine='text-embedding-ada-002') if x else None
            )
            df.to_pickle(proj_dir_pikl)
            logger.info(f"Embeddings saved to {proj_dir_pikl}")
            while True:
                ask = input(f"\n{TerminalColors.OKCYAN}USER:{TerminalColors.ENDC} ")
                result = df_search_sum(df, ask, n=n, n_lines=context)
                chatbot(df, f"Context from embeddings: {result}\nUSER: {ask}")

    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
