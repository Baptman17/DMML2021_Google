import pandas as pd

def get_training_data():
    url = "https://raw.githubusercontent.com/Baptman17/DMML2021_Google/main/data/training_data.csv"
    return pd.read_csv(url, delimiter=",")

def get_unlabelled_test_data():
    url = "https://raw.githubusercontent.com/Baptman17/DMML2021_Google/main/data/unlabelled_test_data.csv"
    return pd.read_csv(url, delimiter=",")