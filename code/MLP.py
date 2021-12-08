from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from util import get_training_data, get_tfidf_vector, configs, get_unlabelled_test_data
from sklearn.metrics import accuracy_score
import pandas as pd


print("Getting data...")
df = get_training_data()
df["difficulty"] = df["difficulty"].astype("category")
print("Preparing ")


print("Splitting data...")
X = df['sentence']
y = df['difficulty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

result = []
res = 0
res2 = 0
# best with tanh and sgd conf no 6
for config in configs():
    for activation in ["tanh"]: #["identity", "logistic", "tanh", "relu"]:
        for solver in ["sgd"]: #["lbfgs", "sgd", "adam"]:
            tfidf_vector = get_tfidf_vector(config)
            classifier = MLPClassifier(max_iter=10000,
                                       activation=activation,
                                       solver=solver)

            pipe = Pipeline([('vectorizer', tfidf_vector),
                             ('classifier', classifier)])

            #print("Fitting the model...")
            pipe.fit(X_train, y_train)

            #print("Predicting the values...")
            y_pred = pipe.predict(X_test)

            #print("Evaluating the prediction...")
            #result.append([config, accuracy_score(y_test, y_pred)])
            res = accuracy_score(y_test, y_pred)
            print(f"{activation}\t{solver}\t{res}")
            result.append(res)
            if(res > res2):
                res2 = res
                df_test = get_unlabelled_test_data()
                y_pred_test = pipe.predict(df_test['sentence'])
                df_final = pd.DataFrame()
                df_final['id'] = list(range(0, len(df_test)))
                df_final['difficulty'] = y_pred_test
                df_final.to_csv("mlp"+ str(res) +".csv", index = False)

print(result)
