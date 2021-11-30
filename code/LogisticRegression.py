from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from util import get_training_data, get_tfidf_vector, evaluate
import pandas as pd

print("Getting data...")
df = get_training_data()
df["difficulty"] = df["difficulty"].astype("category")
print(df)
print("Preparing ")


print("Splitting data...")
X = df['sentence']
y = df['difficulty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

classifier = LogisticRegression()

pipe = Pipeline([('vectorizer', get_tfidf_vector()),
                 ('classifier', classifier)])

print("Fitting the model...")
pipe.fit(X_train, y_train)

print("Predicting the values...")
y_pred = pipe.predict(X_test)

print("Evaluating the prediction...")
evaluate(y_test, y_pred)
