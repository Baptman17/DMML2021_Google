from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from util import get_training_data, get_tfidf_vector, evaluate, get_unlabelled_test_data
import pandas as pd
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import spacy
import syllables
import string
import re

print("Getting data...")
df = get_training_data()
df_test = get_unlabelled_test_data()
df["difficulty"] = df["difficulty"].astype("category")

print("Preparing...")
sp = spacy.load('fr_core_news_sm')
stop_words = list(fr_stop) + list(en_stop)
punctuations = string.punctuation

tab = []
tab_test = []

i = 0
tab_avg_length = []
tab_freq_stop = []
tab_gunning = []
tab_pv = []
tab_cir = []

while i < len(df['sentence']):
    tab_pv.append(len(df['sentence'][i].split(';')))
    cmpt = 0
    cmptacc = 0
    words = df['sentence'][i].translate(str.maketrans('','',string.punctuation)).split(' ')
    for word in words:
        cmpt = cmpt + len(word)
        if('î' in word or 'û' in word or 'ô' in word):
            cmptacc = cmptacc + 1
    tab_avg_length.append(cmpt/len(words))
    tab_cir.append(cmptacc/len(words)) 
    i = i + 1

df['avg_length'] = tab_avg_length
df['pv'] = tab_pv
df['cir'] = tab_cir
i = 0
tab_avg_length = []
tab_freq_stop = []
tab_gunning = []
tab_pv = []
tab_cir = []

while i < len(df_test['sentence']):

    tab_pv.append(len(df_test['sentence'][i].split(';')))
    cmpt = 0
    cmptacc = 0
    words = df_test['sentence'][i].translate(str.maketrans('','',string.punctuation)).split(' ')
    for word in words:
        cmpt = cmpt + len(word)
        if('î' in word or 'û' in word or 'ô' in word):
            cmptacc = cmptacc + 1
    tab_avg_length.append(cmpt/len(words))
    tab_cir.append(cmptacc/len(words))
    i = i + 1

df_test['avg_length'] = tab_avg_length
df_test['pv'] = tab_pv
df_test['cir'] = tab_cir
print("Splitting data...")
X = df[['avg_length','pv','cir']]
y = df['difficulty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

LR_cv = KNeighborsClassifier(n_neighbors=20)
LR_cv.fit(X_train, y_train)
y_pred = LR_cv.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

y_pred_test = LR_cv.predict(df_test[['avg_length','pv','cir']])
df_final = pd.DataFrame()
df_final['id'] = list(range(0, len(df_test)))
df_final['difficulty'] = y_pred_test
df_final.to_csv("submission_LRpv.csv", index = False)