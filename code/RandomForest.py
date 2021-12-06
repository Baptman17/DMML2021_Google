from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from util import get_unlabelled_test_data
from util import get_training_data, get_tfidf_vector, evaluate
import pandas as pd
from sklearn.base import TransformerMixin
import string
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print("Getting data...")
df = get_training_data()
df_test = get_unlabelled_test_data()
df["difficulty"] = df["difficulty"].astype("category")

print("Splitting data...")
X = df['sentence']
y = df['difficulty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

pipe = Pipeline([('vectorizer', get_tfidf_vector()), ('random_forest', RandomForestClassifier())], verbose = True)

pipe.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pipe.predict(X_test)))