from numpy import vectorize
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from util import get_training_data, get_tfidf_vector, evaluate, configs, get_unlabelled_test_data
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


print("Getting data...")
df = get_training_data()
df["difficulty"] = df["difficulty"].astype("category")
print(df)
print("Preparing ")


print("Splitting data...")
X = df['sentence']
y = df['difficulty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

result = []
res = 0
res2 = 0
for config in configs():
    tfidf_vector = get_tfidf_vector(config)
    classifier = KNeighborsClassifier(n_neighbors=49)

    pipe = Pipeline([('vectorizer', tfidf_vector),
                     ('classifier', classifier)])

    print("Fitting the model...")
    pipe.fit(X_train, y_train)

    print("Predicting the values...")
    y_pred = pipe.predict(X_test)

    print("Evaluating the prediction...")
    result.append([config, accuracy_score(y_test, y_pred)])
    res = accuracy_score(y_test, y_pred)
    if(res > res2):
        res2 = res
        df_test = get_unlabelled_test_data()
        y_pred_test = pipe.predict(df_test['sentence'])
        df_final = pd.DataFrame()
        df_final['id'] = list(range(0, len(df_test)))
        df_final['difficulty'] = y_pred_test
        df_final.to_csv("sub_KNN"+ str(res) +".csv", index = False)

print(result)
