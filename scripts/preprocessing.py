import re

import numpy as np
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
random.seed(42)


def shuffle_string(x: str) -> str:
    s = x
    start, end = s[0], s[-1]
    s = list(s[1:-1])
    random.shuffle(s)
    s = ''.join(s)
    s = start + s + end
    return s


def get_typoglycemia_modified_data(df: pd.DataFrame) -> pd.DataFrame:
    typo_easy = []
    typo_hard = []

    for idx, row in df.iterrows():
        easy = row['Easy'].replace('.', '').split(' ')
        hard = row['Hard'].replace('.', '').split(' ')

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

def sentence_preproces(x:str) -> list:
    #Remove all chars that is not a full stop, space or in the alphabet
    x = re.sub('[^a-zA-Z\s\.]', '', x)
    #Remove multiple dots
    x = re.sub('\.{2,}', ' ', x)
    #Remove . in acronymns
    x = re.sub(r'\b([a-zA-Z]\.){2,}[a-zA-Z]\b', lambda y: y.group().replace('.', ''), x)
    #Remove any lenght of spaces except 1
    x = x = re.sub('\s{2,}', ' ', x)
    return x.strip()


def char_to_index(char):
    if 'a' <= char <= 'z':
        return (ord(char) - ord('a') + 1)
    if char == " ":
        # return ord(char)
        return 26
    else:
        return 0
        
def convert_sentence_to_char_sequence(sentences: pd.Series, max_length: int, target: bool) -> torch.Tensor:

    sequences = np.zeros((len(sentences), max_length), dtype= np.float32) - 1
    
    #If target keep it as a categorical value (int)
    if target:
        sequences = np.zeros((len(sentences), max_length)) - 1

    for sentence_idx, sentence in enumerate(sentences):
        for char_idx, char in enumerate(sentence):
            if char_idx < max_length:
                sequences[sentence_idx, char_idx] = char_to_index(char.lower())
            else:
                break
    
    #If not target, make it a float
    if target == False:
        sequences = sequences/100

    return torch.Tensor(sequences)

def tokenize_dataframe(df: pd.DataFrame, complexity: str) -> pd.DataFrame:
    df.loc[:, complexity] = df[complexity].apply(lambda x: ' '.join(sentence_tokennizer(x)))
    df.loc[:, complexity + "_Typo"] = df[complexity + "_Typo"].apply(lambda x: ' '.join(sentence_tokennizer(x)))
    return df

def get_max_length(df: pd.DataFrame, complexity_level: str):
    # Combine the relevant sentence columns
    all_sentences = pd.concat([df[complexity_level], df[complexity_level + "_Typo"]])

    lengths = all_sentences.str.len()

    # Calculate statistics
    max_length = lengths.max()
    mean_length = lengths.mean()
    std_length = lengths.std()
    median_length = lengths.median()

    # Calculate the five-number summary
    min_length = lengths.min()
    q1_length = lengths.quantile(0.25)  # First quartile
    q3_length = lengths.quantile(0.75)  # Third quartile

    # Print the five-number summary
    print(
        f"Five-number summary: Min: {min_length}, Q1: {q1_length}, Median: {median_length}, Q3: {q3_length}, Max: {max_length}")
    print(f"Mean: {mean_length}, Std Dev: {std_length}")

    # Plot the distribution of lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=30, kde=True, color='blue', stat='density', alpha=0.6)
    plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_length:.2f}')
    plt.axvline(median_length, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_length:.2f}')
    plt.axvline(q1_length, color='orange', linestyle='dashed', linewidth=1, label=f'Q1: {q1_length:.2f}')
    plt.axvline(q3_length, color='purple', linestyle='dashed', linewidth=1, label=f'Q3: {q3_length:.2f}')

    plt.title('Distribution of Sentence Lengths')
    plt.xlabel('Length of Sentences')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    return max_length


if __name__ == "__main__":
    sentences = pd.Series(["Hello world", "test sentence"])
    # sentences = pd.Series(["Hello world"])
    tensor_output = convert_sentence_to_char_sequence(sentences, 30)
    print(tensor_output)
