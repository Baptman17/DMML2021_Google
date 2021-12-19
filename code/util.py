import pandas as pd
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import  TfidfVectorizer
import string
from EvaluationMetrics import EvaluationMetrics

stop_words = spacy.lang.fr.stop_words.STOP_WORDS
sp = spacy.load('fr_core_news_sm')
punctuations = string.punctuation
numbers = "0123456789"


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
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Remove stop words and punctuation
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # Remove anonymous dates and people
    mytokens = [ word.replace('xx/', '').replace('xxxx/', '').replace('xx', '') for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in ["xxxx", "xx", ""]]

    # Remove sufix like ".[1" in "experience.[1"
    mytokens_2 = []
    for word in mytokens:
      for char in word:
        if (char in punctuations) or (char in numbers):
          word = word.replace(char, "")
      if word != "":
        mytokens_2.append(word)

    # Return preprocessed list of tokens
    return mytokens_2

def spacy_tokenizer_without_dc(sentence):
    # Create token object, which is used to create documents with linguistic annotations.
    mytokens = sp(sentence)

    mytokens = [ word.lemma_.lower().strip() for word in mytokens ]

    # Return preprocessed list of tokens
    return mytokens


def get_tfidf_vector(config=None, with_dc: bool = False):
    if with_dc:
        return TfidfVectorizer(tokenizer=spacy_tokenizer,
                                       ngram_range=config[0],
                                       min_df=config[1], max_df=config[2], analyzer=config[3])
    else:
        return TfidfVectorizer(tokenizer=spacy_tokenizer_without_dc)

def configs():
    models = list()
    # Define config lists
    ngram_range = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    min_df = [1]
    max_df = [1.0]
    analyzer = ['word', 'char']
    # Create config instances
    for n in ngram_range:
        for i in min_df:
            for j in max_df:
                for a in analyzer:
                    cfg = [n, i, j, a]
                    models.append(cfg)
    return models


def evaluate(true, pred):
    return EvaluationMetrics(true, pred)
