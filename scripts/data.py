import re
from num2words import num2words
import pandas as pd
import torch.utils.data
import os
import torch

from scripts import preprocessing
from scripts.preprocessing import convert_sentence_to_char_sequence, get_typoglycemia_modified_data


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def generate_typoglycemia_data_file(similarity_threshold: float, file_path: str):
    df = pd.read_csv(file_path, sep="\t", names=["Hard", "Easy", "Similarity"])
    df = df[df["Similarity"] <= similarity_threshold]
    df['Hard'] = df['Hard'].apply(convert_numbers_to_words)
    df['Easy'] = df['Easy'].apply(convert_numbers_to_words)

    df = get_typoglycemia_modified_data(df)
    df.reset_index(inplace=True, drop=True)
    df.to_csv("../data/processed/sscorpus.csv", index=False)


def convert_numbers_to_words(sentence):
    return re.sub(r'\b\d+\b', lambda x: num2words(int(x.group())), sentence)


def create_data_loader(df: pd.DataFrame, complexity: str = "Easy", batch_size: int = 16) -> torch.utils.data.DataLoader:
    df[complexity] = df[complexity].apply(lambda x: ' '.join(preprocessing.sentence_tokennizer(x)))
    df[complexity + "_Typo"] = df[complexity + "_Typo"].apply(lambda x: ' '.join(preprocessing.sentence_tokennizer(x)))

    # Combine both columns into one series with all sentences
    all_sentences = pd.concat([df[complexity], df[complexity + "_Typo"]])

    # Get the length of the longest sentence
    max_length = all_sentences.str.len().max()

    sentence = convert_sentence_to_char_sequence(df[complexity], max_length)
    typo_sentence = convert_sentence_to_char_sequence(df[complexity + "_Typo"], max_length)

    dataset = DataClass(typo_sentence, sentence)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


class DataClass(torch.utils.data.Dataset):
    def __init__(self, typo_sentence, target_sentence):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.typo_sentence = typo_sentence.to(device)
        self.target_sentence = target_sentence.to(device)

    def __len__(self):
        return self.typo_sentence.shape[0]

    def __getitem__(self, idx):
        return self.typo_sentence[idx], self.target_sentence[idx]


if __name__ == "__main__":
    generate_typoglycemia_data_file(similarity_threshold=0.7, file_path="../data/raw/sscorpus")

