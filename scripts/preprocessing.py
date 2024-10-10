import pandas as pd
import torch
import random
random.seed(42)

def shuffle_string(x:str) -> str:
    #typo = x
    #print(x)
    #while typo == x: #To ensure that it does not return the same word
    s = x
    start, end = s[0], s[-1]
    s = list(s[1:-1])
    random.shuffle(s)
    s = ''.join(s)
    s = start + s + end
    #typo = s
    return s

def get_typoglycemia_modified_data(df: pd.DataFrame) -> pd.DataFrame:
    typo_easy = []
    typo_hard = []

    for idx, row in df.iterrows():
        #print(idx)
        easy = row['Easy'].split(' ')
        hard = row['Hard'].split(' ')

        #shuffle words
        easy = [shuffle_string(i) if len(i) > 3 else i for i in easy]
        hard = [shuffle_string(i) if len(i) > 3 else i for i in hard]
        
        typo_easy.append(' '.join(easy))
        typo_hard.append(' '.join(hard))

    df['Easy_Typo'] = typo_easy
    df['Hard_Typo'] = typo_easy

    return df


def convert_sentence_to_char_sequence(df: pd.DataFrame) -> torch.Tensor:  # TODO
    return None
