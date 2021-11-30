from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from util import get_training_data, get_tfidf_vector, evaluate, get_unlabelled_test_data
import pandas as pd
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import spacy
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
while i < len(tab):
    df['sentence'][i] = ' '.join(tab[i])
    i = i + 1

for line in df_test['sentence']:
    # Create token object, which is used to create documents with linguistic annotations.
    mytokens = sp(line)

    # Lemmatize each token and convert each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in mytokens ]
    # Remove stop words and punctuation
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    tab_test.append(mytokens)
i = 0
while i < len(tab_test):
    df_test['sentence'][i] = ' '.join(tab_test[i])
    i = i + 1

print("Splitting data...")
X = df['sentence']
y = df['difficulty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

pipe = Pipeline([('vectorizer', get_tfidf_vector()), ('decision_tree', DecisionTreeClassifier())], verbose = True)

pipe.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pipe.predict(X_test)))

#0.35104

y_pred_test = pipe.predict(df_test['sentence'])
df_final = pd.DataFrame()
df_final['id'] = list(range(0, len(df_test)))
df_final['difficulty'] = y_pred_test
df_final.to_csv("submission_DecisionTree.csv", index = False)