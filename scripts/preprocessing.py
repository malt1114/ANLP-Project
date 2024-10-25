import re

import numpy as np
import pandas as pd
import torch
import random

random.seed(42)


def shuffle_string(x: str) -> str:
    # typo = x
    # print(x)
    # while typo == x: #To ensure that it does not return the same word
    s = x
    start, end = s[0], s[-1]
    s = list(s[1:-1])
    random.shuffle(s)
    s = ''.join(s)
    s = start + s + end
    # typo = s
    return s


def get_typoglycemia_modified_data(df: pd.DataFrame) -> pd.DataFrame:
    typo_easy = []
    typo_hard = []

    for idx, row in df.iterrows():
        # print(idx)
        easy = row['Easy'].split(' ')
        hard = row['Hard'].split(' ')

        # shuffle words
        easy = [shuffle_string(i) if len(i) > 3 else i for i in easy]
        hard = [shuffle_string(i) if len(i) > 3 else i for i in hard]

        typo_easy.append(' '.join(easy))
        typo_hard.append(' '.join(hard))

    df['Easy_Typo'] = typo_easy
    df['Hard_Typo'] = typo_hard

    return df


def sentence_tokennizer(sentence: str) -> list:
    # Remove all non-alphabet chars
    regex = re.compile('[^a-zA-Z ]')
    sentence = regex.sub('', sentence)
    sentence = sentence.lower()
    sentence = sentence.split(' ')
    # Remove empty strings
    sentence = [i for i in sentence if len(i) != 0]
    return sentence


def convert_sentence_to_char_sequence(sentences: pd.Series, max_length: int) -> torch.Tensor:

    sequences = np.zeros((len(sentences), max_length)) - 1

    def char_to_index(char):
        if 'a' <= char <= 'z':
            return ord(char) - ord('a') + 1
        if char == " ":
            return ord(char)
        else:
            return 0

    for sentence_idx, sentence in enumerate(sentences):
        for char_idx, char in enumerate(sentence):
            # print(char, ord(char))
            sequences[sentence_idx, char_idx] = char_to_index(char.lower())

    return torch.Tensor(sequences)



if __name__ == "__main__":
    sentences = pd.Series(["Hello world", "test sentence"])
    # sentences = pd.Series(["Hello world"])
    tensor_output = convert_sentence_to_char_sequence(sentences, 30)
    print(tensor_output)
