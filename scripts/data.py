import re
from num2words import num2words
import pandas as pd
import torch.utils.data
import os
import torch
from pycodestyle import maximum_doc_length
from scripts.model  import device
from scripts import preprocessing
from scripts.preprocessing import convert_sentence_to_char_sequence, get_typoglycemia_modified_data, sentence_preproces


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def generate_typoglycemia_data_file(similarity_threshold: float, file_path: str):
    df = pd.read_csv(file_path, sep="\t", names=["Hard", "Easy", "Similarity"])
    print(df.shape)
    df = df[df["Similarity"] <= similarity_threshold]
    print(df.shape)

    #Split sentence at full stop and clean the sentences
    df['Hard'] = df['Hard'].apply(sentence_preproces)
    df['Easy'] = df['Easy'].apply(sentence_preproces)

    df['Hard'] = df['Hard'].apply(convert_numbers_to_words)
    df['Easy'] = df['Easy'].apply(convert_numbers_to_words)

    df = get_typoglycemia_modified_data(df)
    df.reset_index(inplace=True, drop=True)
    df.to_csv("data/processed/sscorpus.csv", index=False)


def convert_numbers_to_words(sentence):
    return re.sub(r'\b\d+\b', lambda x: num2words(int(x.group())), sentence)


def create_data_loader(df: pd.DataFrame, complexity: str = "Easy", batch_size: int = None, max_length: int = None) -> torch.utils.data.DataLoader:
    # sentence = convert_sentence_to_char_sequence(df[complexity], max_length).view(-1, batch_size * max_length)
    # typo_sentence = convert_sentence_to_char_sequence(df[complexity + "_Typo"], max_length).reshape(-1, max_length, 1)
    sentence = convert_sentence_to_char_sequence(df[complexity], max_length, target = True)
    typo_sentence = convert_sentence_to_char_sequence(df[complexity + "_Typo"], max_length, target = False)
    dataset = DataClass(typo_sentence, sentence, max_length, batch_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader






class DataClass(torch.utils.data.Dataset):
    def __init__(self, typo_sentence, target_sentence, max_length, batch_size):
        self.typo_sentence = typo_sentence.to(device)
        self.target_sentence = target_sentence.long().to(device)
        self.max_length = max_length
        self.batch_size = batch_size
    def __len__(self):
        return self.typo_sentence.shape[0]

    def __getitem__(self, idx):
        return self.typo_sentence[idx], self.target_sentence[idx]



if __name__ == "__main__":
    generate_typoglycemia_data_file(similarity_threshold=0.7, file_path="../data/raw/sscorpus")

