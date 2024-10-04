import pandas as pd
import torch.utils.data
import os


def load_data(similarity_threshold: float) -> pd.DataFrame:
    path = "data/raw/sscorpus/parawiki_english05"
    df = pd.read_csv(path, sep="\t", names=["Hard", "Easy", "Similarity"])
    df = df[df["Similarity"] <= similarity_threshold]
    return df


def create_data_loader(df: pd.DataFrame) -> torch.utils.data.DataLoader | None:

    return None


if __name__ == "__main__":
    os.chdir('..')
    data_dict = load_data(0.8)
    print(data_dict)
