import re
import pandas as pd
import torch.utils.data
import os
import torch
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

    for t in ['Hard', 'Easy']:
        t_df = get_typoglycemia_modified_data(df, level = t)
        t_df.reset_index(inplace=True, drop=True)
        t_df.to_csv(f"data/processed/{t.lower()}/sscorpus_{t.lower()}.csv", index=False)


def create_data_loader(df: pd.DataFrame, complexity: str = "Easy", batch_size: int = None, max_length: int = None) -> torch.utils.data.DataLoader:
    sentence = convert_sentence_to_char_sequence(df[complexity], max_length, target = True)
    typo_sentence = convert_sentence_to_char_sequence(df['typoglycemia'], max_length, target = False)
    dataset = DataClass(typo_sentence, sentence, max_length, batch_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataClass(torch.utils.data.Dataset):
    def __init__(self, typo_sentence, target_sentence, max_length, batch_size):
        self.typo_sentence = typo_sentence
        self.target_sentence = target_sentence.long()
        self.max_length = max_length
        self.batch_size = batch_size
    def __len__(self):
        return self.typo_sentence.shape[0]

    def __getitem__(self, idx):
        return self.typo_sentence[idx], self.target_sentence[idx]



if __name__ == "__main__":
    generate_typoglycemia_data_file(similarity_threshold=0.7, file_path="../data/raw/sscorpus")