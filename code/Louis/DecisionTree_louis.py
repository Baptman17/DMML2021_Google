from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from util import get_training_data, get_tfidf_vector, evaluate, get_unlabelled_test_data
import pandas as pd

print("Getting data...")
df = get_training_data()
df_test = get_unlabelled_test_data()
df["difficulty"] = df["difficulty"].astype("category")
print("Preparing ")


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