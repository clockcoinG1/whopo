````
def construct_prompt(self, question: str) -> str:
    encoding = tiktoken.get_encoding(self.ENCODING)
    separator_len = len(encoding.encode(self.SEPARATOR))

    relevant_code = self.search_functions(df=self.df, code_query=question, n=5)
    chosen_sections = []
    chosen_sections_len = 0

    for _, row in relevant_code.iterrows():
        code_str = f"File: {row['file_name']}\n" \
                   f"Path: {row['file_path']}\n" \
                   f"Similarity: {row['similarities']}\n" \
                   f"Lines of Code: {row['lines_of_code']}\n" \
                   f"Code:\n{row['code']}\n"

        code_str_len = len(encoding.encode(code_str))



        if chosen_sections_len + separator_len + code_str_len > self.MAX_SECTION_LEN:
            break

        chosen_sections.append(self.SEPARATOR + code_str)
        chosen_sections_len += separator_len + code_str_len

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    chosen_sections_str = "".join(chosen_sections)
    print(chosen_sections_str)

    return f'\nCode base context from embeddings dataframe:\n\n{chosen_sections_str}<|/knowledge|>'
USER: Lets think outside the box. How can we make this code more robust?
ASSISTANT: