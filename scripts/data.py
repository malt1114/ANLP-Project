import pandas as pd
import torch.utils.data
import os
import torch
from scripts.preprocessing import convert_sentence_to_char_sequence

def load_data(similarity_threshold: float, file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", names=["Hard", "Easy", "Similarity"])
    df = df[df["Similarity"] <= similarity_threshold]
    df = df.drop_duplicates('Hard')
    df = df.drop_duplicates('Easy')
    return df


def create_data_loader(df: pd.DataFrame, complexity: str = "Easy", batch_size: int = 16) -> torch.utils.data.DataLoader:
    sentence = convert_sentence_to_char_sequence(df[complexity])
    typo_sentence = convert_sentence_to_char_sequence(df["typo_"+complexity])

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