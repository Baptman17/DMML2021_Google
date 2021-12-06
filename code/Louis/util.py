import pandas as pd
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from spacy.lang.fr.stop_words import STOP_WORDS
from spacy.lang.fr import French
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import numpy as np

stop_words = spacy.lang.fr.stop_words.STOP_WORDS
stop_words_en = spacy.lang.en.stop_words.STOP_WORDS
sp = spacy.load('fr_core_news_sm')
sp2 = spacy.load("en_core_web_sm")
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

def english_token(sentence):
    # Create token object, which is used to create documents with linguistic annotations.
    mytokens = sp2(sentence)

    # Lemmatize each token and convert each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in mytokens ]

    # Remove stop words and punctuation
    mytokens = [ word for word in mytokens if word in stop_words_en and word not in punctuations ]

    # Return preprocessed list of tokens
    return mytokens

def spacy_tokenizer2(sentence):
    nlp = spacy.load('fr_core_news_sm')
    with nlp.disable_pipes():
        vectors = np.array([token.vector for token in  nlp(sentence)])
    return vectors

def get_tfidf_vector():
    return TfidfVectorizer(tokenizer=spacy_tokenizer)

def get_tfidf_vector2():
    return TfidfVectorizer(tokenizer=spacy_tokenizer2)


def evaluate(true, pred):
    precision = precision_score(true, pred, average='micro')
    recall = recall_score(true, pred, average='micro')
    f1 = f1_score(true, pred, average='micro')
    print(f"CONFUSION MATRIX:\n{confusion_matrix(true, pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(true, pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n\tPrecision: {precision:.4f}\n\tRecall: {recall:.4f}\n\tF1_Score: {f1:.4f}")