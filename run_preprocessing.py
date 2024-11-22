from scripts.data import generate_typoglycemia_data_file, load_data
from sklearn.model_selection import train_test_split

generate_typoglycemia_data_file(similarity_threshold = 0.7, file_path = "data/raw/sscorpus")

for i in ['Easy', 'Hard']:
    df = load_data(file_path = f"data/processed/{i.lower()}/sscorpus_{i.lower()}.csv")
    df = df[(~df['typoglycemia'].isna())]

    #Select sentences of lenght X
    df['len'] = df['typoglycemia'].apply(len)
    df = df[df['len'] <= 150]
    df = df[[i, 'typoglycemia']]

    #Create train, test val and save as files
    dev, test = train_test_split(df, test_size=0.2, shuffle= True)
    train, validation = train_test_split(dev, test_size=0.2, shuffle= True)
    
    test.reset_index(inplace=True, drop=True)
    test.to_csv(f"data/processed/{i.lower()}/test_{i.lower()}.csv", index=False)

    train.reset_index(inplace=True, drop=True)
    train.to_csv(f"data/processed/{i.lower()}/train_{i.lower()}.csv", index=False)

    validation.reset_index(inplace=True, drop=True)
    validation.to_csv(f"data/processed/{i.lower()}/validation_{i.lower()}.csv", index=False)


