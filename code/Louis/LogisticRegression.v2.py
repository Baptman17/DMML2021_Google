from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from util import get_training_data, get_tfidf_vector, evaluate, get_unlabelled_test_data
import pandas as pd
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import spacy
import syllables
import string

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

for line in df['sentence']:
    # Create token object, which is used to create documents with linguistic annotations.
    mytokens = sp(line)
    # Lemmatize each token and convert each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in mytokens ]
    # Remove stop words and punctuation
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    tab.append(mytokens)
i = 0
tab_avg_length = []
tab_freq_stop = []
tab_gunning = []

while i < len(tab):
    cmpt = 0
    cmpt3syl = 0
    for word in tab[i]:
        cmpt = cmpt + len(word)
        if(syllables.estimate(word) > 2):
            cmpt3syl = cmpt3syl + 1
    tab_avg_length.append(cmpt/(len(tab[i])+1))
    tab_word = df['sentence'][i].split()
    tab_freq_stop.append(len(tab[i])/len(tab_word))
    sentences = df['sentence'][i].split('.')[:-1]
    avg_sent_length = len(tab_word)/(len(sentences)+1)
    tab_gunning.append(0.4*(avg_sent_length + 100*(cmpt3syl/len(tab_word))))
    i = i + 1
df['freq_stopwords'] = tab_freq_stop
df['avg_length'] = tab_avg_length
df['gunning'] = tab_gunning

for line in df_test['sentence']:
    # Create token object, which is used to create documents with linguistic annotations.
    mytokens = sp(line)

    # Lemmatize each token and convert each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in mytokens ]
    # Remove stop words and punctuation
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    tab_test.append(mytokens)
i = 0
tab_avg_length = []
tab_freq_stop = []
tab_gunning = []

while i < len(tab_test):
    cmpt = 0
    for word in tab_test[i]:
        cmpt = cmpt + len(word)
    if(syllables.estimate(word) > 2):
            cmpt3syl = cmpt3syl + 1
    tab_avg_length.append(cmpt/(len(tab_test[i])+1))
    tab_word = df_test['sentence'][i].split()
    tab_freq_stop.append(len(tab_test[i])/len(tab_word))
    sentences = df_test['sentence'][i].split('.')[:-1]
    avg_sent_length = len(tab_word)/(len(sentences)+1)
    tab_gunning.append(0.4*(avg_sent_length + 100*(cmpt3syl/len(tab_word))))
    i = i + 1
df_test['freq_stopwords'] = tab_freq_stop
df_test['avg_length'] = tab_avg_length
df_test['gunning'] = tab_gunning

print("Splitting data...")
X = df[['freq_stopwords','avg_length','gunning']]
#03927
#X = df[['gunning']]
#0.3020833333333333
#X = df[['freq_stopwords','gunning']]
#0.328125
y = df['difficulty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

LR_cv = LogisticRegressionCV(solver='lbfgs', cv=5, max_iter=1000, random_state=72)
LR_cv.fit(X_train, y_train)
y_pred = LR_cv.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

y_pred_test = LR_cv.predict(df_test[['freq_stopwords','avg_length','gunning']])
df_final = pd.DataFrame()
df_final['id'] = list(range(0, len(df_test)))
df_final['difficulty'] = y_pred_test
df_final.to_csv("submission_DecisionTree.csv", index = False)