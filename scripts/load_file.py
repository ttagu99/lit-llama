import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm

# %%
# 파일 확장자를 추출하는 함수
def extract_extension(file_path):
    return str(file_path).split('.')[-1]

# %%
# 파일 확장자가 json인지 jsonl인지 구분하여 Load
def load_file(file_path) -> None:
    with open(file_path, "r") as file:  
        if extract_extension(file_path)== 'json':
            data = json.load(file)
        elif extract_extension(file_path)== 'jsonl':
            data = [json.load(line) for line in file]
        else:
            print('파일 확장자를 확인해주세요.')
        print("total data: ", len(data))
    return data


# %%
