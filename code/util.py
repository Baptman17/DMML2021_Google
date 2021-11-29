import pandas as pd
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from spacy.lang.fr import French

stop_words = spacy.lang.fr.stop_words.STOP_WORDS
sp = spacy.load('fr_core_web_sm')

def get_training_data():
    url = "https://raw.githubusercontent.com/Baptman17/DMML2021_Google/main/data/training_data.csv"
    return pd.read_csv(url, delimiter=",")


def get_unlabelled_test_data():
    url = "https://raw.githubusercontent.com/Baptman17/DMML2021_Google/main/data/unlabelled_test_data.csv"
    return pd.read_csv(url, delimiter=",")


def spacy_tokenizer(sentence):
    # Create token object, which is used to create documents with linguistic annotations.
    mytokens = sp(sentence)

    # Lemmatize each token and convert each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in mytokens ]
    ## alternative way
    # mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Remove stop words and punctuation
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # Return preprocessed list of tokens
    return mytokens