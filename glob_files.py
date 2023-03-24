import os
from glob import glob
import pandas as pd


import os
from glob import glob
import pandas as pd


def glob_files(directory, extensions):
    all_files = []
    for ext in extensions.split(','):
        all_files.extend(glob(os.path.join(directory, "**", f"*.{ext}"), recursive=True))

    result = []
    for f in all_files:
        file_name = os.path.basename(f)
        file_path = f
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        try:
            with open(f, "r", errors='ignore') as file:
                code = file.read()
        except IsADirectoryError:
            print(f"Skipping directory: {file_path}")
            continue
        except Exception as e:
            print(f"Error while reading file {file_path}: {e}")
            continue

        result.append({
            "file_name": file_name,
            "file_path": file_path,
            "code": code
        })

    result = pd.DataFrame(result)
    return result
