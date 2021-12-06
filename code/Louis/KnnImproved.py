from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from util import english_token
from util import spacy_tokenizer
from util import get_unlabelled_test_data
from util import get_training_data, get_tfidf_vector, evaluate
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier
import string
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
import re
import syllables
import readability

print("Getting data...")
stop_words = list(fr_stop)
df = get_training_data()
df_test = get_unlabelled_test_data()
df["difficulty"] = df["difficulty"].astype("category")
i = 0
avg_len_word = []
avg_stop = []
avg_word_per_sent = []
nb_total_words = []
nb_total_sentence = []
tab_freq_complex = []
tab4syl = []
tab3syl = []
tab_en_stop = []
while i < len(df['sentence']):
    tab_en_stop.append(len(english_token(df['sentence'][i])))
    tokens = spacy_tokenizer(df['sentence'][i])
    cmpt3syl = 0
    cmpt4syl = 0
    for token in tokens:
        if(syllables.estimate(token) > 3):
            cmpt4syl = cmpt4syl + 1
        elif(syllables.estimate(token) > 2):
            cmpt3syl = cmpt3syl + 1
    tab4syl.append(cmpt4syl)
    tab3syl.append(cmpt3syl)
    words = df['sentence'][i].split()
    tab_freq_complex.append(len(tokens)/len(words))
    nb_total_words.append(len(words))
    df['sentence'][i] = df['sentence'][i].replace('!','.').replace('?','.').replace('...','.')
    sentences = df['sentence'][i].split('.')
    nb_total_sentence = len(sentences)
    cmptword = 0
    for sentence in sentences:
        words_sent = sentence.split()
        cmptword = cmptword + len(words_sent)
    avg_word_per_sent.append(cmptword/len(sentences))
    tab_avg_len = 0
    cmpt_stop = 0
    for word in words:
        if(word in stop_words):
            cmpt_stop = cmpt_stop + 1
        if('...' in word):
            tab_avg_len = tab_avg_len + len(word)-3
        elif(',' in word or '.' in word or '!' in word or '?' in word):
            tab_avg_len = tab_avg_len + len(word)-1
        else:
            tab_avg_len = tab_avg_len + len(word)
    avg_stop.append(cmpt_stop/len(words))
    avg_len_word.append(tab_avg_len/len(words))
    i = i + 1
df['avg_len_word'] = avg_len_word
df['avg_stop'] = avg_stop
df['avg_word_per_sent'] = avg_word_per_sent
df['nb_words'] = nb_total_words
df['nb_total_sentences'] = nb_total_sentence
df['avg_word*nb_words'] = np.multiply(avg_word_per_sent,nb_total_words)
df['freq_complex'] = tab_freq_complex
df['3syl'] = tab3syl
df['4syl'] = tab4syl
df['eng_stop'] = tab_en_stop
print(df.iloc[200:250])
print("Splitting data...")
X = df[['avg_len_word','avg_word_per_sent','3syl','4syl','eng_stop']]
y = df['difficulty']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234, stratify=y)

knnclass = KNeighborsClassifier(n_neighbors=5) 

knnclass.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, knnclass.predict(X_test)))