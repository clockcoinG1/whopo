import argparse
import os
import sys
import pandas as pd
import re
from embedder import CodeExtractor
from chatbot import indexCodebase, df_search_sum, chatbot, generate_summary, write_md_files

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Code summarization chatbot')
    parser.add_argument('--directory', type=str, default="/ezcoder", help='directory to summarize')
    parser.add_argument('-P', type=str, default="", help='saved db to use ')
    parser.add_argument('--root', type=str, default=f"{os.environ['CODE_EXTRACTOR_DIR']}", help='Where root of project is or env $CODE_EXTRACTOR_DIR')
    parser.add_argument('-n', type=int, default=10, help='number of context chunks to use')
    parser.add_argument('--prompt', type=str, default='What does this code do?', help='gpt prompt')
    parser.add_argument('--chat', type=bool, default=True, help='gpt chat')
    parser.add_argument('--context', type=int, default=10, help='context length')
    parser.add_argument('--max_tokens', type=int, default=1000, help='maximum number of tokens in summary')

    # Parse arguments
    args = parser.parse_args()
    proj_dir = args.directory.strip()
    root_dir = args.root.strip()
    prompt = args.prompt.strip()
    n = args.n
    context = args.context
    max_tokens = args.max_tokens

    # Check if saved db is provided
    if args.P:
        if not os.path.isfile(args.P):
            parser.error(f"The file specified does not exist.{args.P}")
        df = pd.read_pickle(args.P)
    else:
        print(f"\033[1;32;40m\nSummarizing {args.directory}\nUsing {args.n} context chunks\nPrompt: {args.prompt}")
        if not os.path.exists(root_dir + "/" + proj_dir):
            print(f"Directory {root_dir + args.directory} does not exist")
            sys.exit()

        # Extract code and generate summary
        ce = CodeExtractor(f"{root_dir}/{proj_dir}")
        df = ce.get_files_df()
        df = ce.split_code_by_lines(df, max_lines=20)
        df = indexCodebase(df, "code", pickle=f"{root_dir}/{proj_dir}.pkl", code_root=f"{root_dir}/{proj_dir}")
        print(f"\033[1;32;40m*" * 20 + "\tGenerating summary...\t" + f"\033[1;32;40m*" * 25)
        df = df[df['code'] != ''].dropna()
        df = generate_summary(df, proj_dir=proj_dir)
        df = df[df['summary'] != ''].dropna()
        print(f"\033[1;32;40m*" * 10 + "\tWriting summary...\t" + f"\033[1;32;40m*" * 10)
        write_md_files(df, f"{proj_dir}".strip('/'))

        # Embed summary and save to pickle file
        proj_dir_pikl = re.sub(r'[^a-zA-Z]', '', f"{root_dir}/{proj_dir}")
        print(f"\033[1;34;40m*" * 20 + "\t Embedding summary column ...\t" + f"{root_dir}/{proj_dir}"  + f"\033[1;34;40m*" * 20)
        df['summary_embedding'] = df['summary'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002') if x else None)
        print(f"\033[1;32;40m*" * 40 + "\t Saving embedding summary...\t" + f"{root_dir}/{proj_dir}"  + f"\033[1;32;40m*" * 40)
        df.to_pickle(proj_dir_pikl + '.pkl')

    # Chat with the chatbot
    if args.chat: 
        while True:
            ask = input("\n\033[33mAsk about the files, code, summaries:\033[0m\n\n\033[44mUSER:  \033[0m")
            summary_items  = df_search_sum(df, ask, pprint=True, n=n , n_lines=context) 
            chatbot(df, f"## context from embedding\nSummaries:\n{summary_items}\n\n USER: {ask}" , context)

if __name__ == '__main__':
    main()