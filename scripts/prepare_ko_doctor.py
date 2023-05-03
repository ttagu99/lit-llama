"""Implementation derived from https://github.com/tloen/alpaca-lora"""
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


DATA_FILE = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"
DATA_FILE_NAME1 = "GenMedGPT-5k_ko.jsonl"
DATA_FILE_NAME2 = "chatdoctor200k_ko.jsonl"
DATA_FILE_NAME3 = "kin_data_2M.jsonl"

IGNORE_INDEX = -1

def extract_extension(file_path):
    return str(file_path).split('.')[-1]

def load_file(file_path) -> None:
    with open(file_path, "r") as file:  
        if extract_extension(file_path)== 'json':
            data = json.load(file)
        elif extract_extension(file_path)== 'jsonl':
            data = []
            for n, line in enumerate(file):
                try:
                    data.append(json.loads(line))
                except:
                    print(f"error line: {n}")
        else:
            print('파일 확장자를 확인해주세요.')
        print("total data: ", len(data))
    return data

def prepare(
    destination_path: Path = Path("data/ko_doctor"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    test_split_size: int = 200,
    max_seq_length: int = 256,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    data_file_name: str = DATA_FILE_NAME1
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path.mkdir(parents=True, exist_ok=True)
    # file_path = destination_path / data_file_name
    # download(file_path)
    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    data1 = load_file(destination_path/DATA_FILE_NAME1)
    print("data1: ", len(data1))
    data2 = load_file(destination_path/DATA_FILE_NAME2)
    print("data2: ", len(data2))
    data3 = load_file(destination_path/DATA_FILE_NAME3)
    print("data3: ", len(data3))
    data = data1 + data2 + data3
    print(f"data length: {len(data)}")
    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data, 
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    torch.save(train_set, destination_path / f"train_{max_seq_length}.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set, destination_path / f"test_{max_seq_length}.pt")


def download(file_path: Path):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w") as f:
        f.write(requests.get(DATA_FILE).text)


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.
    
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""
    if example["input"]:
        return (
            "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
            "요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            f"### 명령어:\n{example['instruction']}\n\n### 입력:\n{example['input']}\n\n### 응답:"
        )
    return (
        "환자가 의사에게 아픈 곳에 대해 문의합니다.\n\n"
        "환자의 문의 내용에 대해 답변하세요. 환자의 질병을 진단하고, 가능하면 처방을 하세요. \n\n"
        f"### 문의:\n{example['instruction']}\n\n### 응답:"
              
        # "아래는 작업을 설명하는 명령어입니다.\n\n"
        # "명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        # f"### 명령어:\n{example['instruction']}\n\n### 응답:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
