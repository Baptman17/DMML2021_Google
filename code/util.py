import pandas as pd
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from spacy.lang.fr.stop_words import STOP_WORDS
from spacy.lang.fr import French
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
from EvaluationMetrics import EvaluationMetrics

stop_words = spacy.lang.fr.stop_words.STOP_WORDS
sp = spacy.load('fr_core_news_sm')
punctuations = string.punctuation


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

    # Remove stop words and punctuation
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # Return preprocessed list of tokens
    return mytokens

def spacy_tokenizer_without_dc(sentence):
    # Create token object, which is used to create documents with linguistic annotations.
    mytokens = sp(sentence)

    # Return preprocessed list of tokens
    return mytokens


def get_tfidf_vector(config):
    return TfidfVectorizer(tokenizer=spacy_tokenizer,
                                   ngram_range=config[0],
                                   min_df=config[1], max_df=config[2], analyzer=config[3])

def get_tfidf_vector_without_dc():
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
    precision = precision_score(true, pred, average='weighted')
    recall = recall_score(true, pred, average='weighted')
    f1 = f1_score(true, pred, average='weighted')
    accuracy = accuracy_score(true, pred)
    return EvaluationMetrics(precision, recall, f1, accuracy)
