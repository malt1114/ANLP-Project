import pandas as pd


def load_data():
    path = "../data/raw/sscorpus/parawiki_english05"
    df = pd.read_csv(path, sep="\t", names=["Hard", "Easy", "Similarity"])
    return df


if __name__ == "__main__":
    data_dict = load_data()
    print(data_dict)
