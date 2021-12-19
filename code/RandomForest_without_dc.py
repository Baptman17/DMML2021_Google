from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from util import get_tfidf_vector_without_dc
import util
from util import get_unlabelled_test_data, get_training_data, get_tfidf_vector, evaluate
import pandas as pd
from sklearn.base import TransformerMixin
import string
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import threading
import time

class RandomForestThread_without_dc(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.__bestMetrics = None

    def run(self):
        self.get_metrics()

    def join(self):
        threading.Thread.join(self)
        return self.__bestMetrics

    def get_metrics(self):
        print("[RF] Getting data")
        start_time = time.time()
        df = get_training_data()
        df["difficulty"] = df["difficulty"].astype("category")

        print("[RF] Splitting data")
        X = df['sentence']
        y = df['difficulty']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

        bestMetrics = None
        config_start_time = time.time()
        tfidf_vector = get_tfidf_vector_without_dc()
        classifier = RandomForestClassifier()

        pipe = Pipeline([('vectorizer', tfidf_vector),
                            ('classifier', classifier)])

        print(f"[RF] Fitting the model")
        pipe.fit(X_train, y_train)

        print(f"[RF] Predicting the values")
        y_pred = pipe.predict(X_test)

        print(f"[RF] Evaluating the prediction")
        metrics = evaluate(y_test, y_pred)
        if bestMetrics is None:
            bestMetrics = metrics
        else:
            if metrics > bestMetrics:
                bestMetrics = metrics
        print(f"[RF] End of evaluation in {time.time() - config_start_time}")
        self.__bestMetrics = bestMetrics
        print(f"[RF] Done in {time.time() - start_time}")