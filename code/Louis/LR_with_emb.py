from numpy import vectorize
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import spacy
from  util import get_tfidf_vector2
from util import get_training_data, get_tfidf_vector, evaluate
import pandas as pd
import numpy as np

print("Getting data...")
df = get_training_data()
df["difficulty"] = df["difficulty"].astype("category")
print(df)
print("Preparing ")


print("Splitting data...")
for sentence in df['sentence'][1:20]:
    nlp = spacy.load('fr_core_news_sm')
    with nlp.disable_pipes():
        vectors = np.array([token.vector for token in  nlp(sentence)])
X = vectors
y = df['difficulty'][1:20]
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

classifier = LogisticRegression()

print("Fitting the model...")
classifier.fit(X_train, y_train)

print("Predicting the values...")
y_pred = classifier.predict(X_test)

print("Evaluating the prediction...")
evaluate(y_test, y_pred)