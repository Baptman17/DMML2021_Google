from numpy import vectorize
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from util import get_tfidf_vector_without_dc

import util
from util import get_training_data, get_tfidf_vector, evaluate, configs, get_unlabelled_test_data
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import threading
import time


class kNNThread_without_dc(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.__bestMetrics = None

    def run(self):
        self.get_metrics()

    def join(self):
        threading.Thread.join(self)
        return self.__bestMetrics

    def get_metrics(self):
        print("[kNN] Getting data")
        start_time = time.time()
        df = get_training_data()
        df["difficulty"] = df["difficulty"].astype("category")
        print("[kNN] Splitting data")
        X = df['sentence']
        y = df['difficulty']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)
        bestMetrics = None
        config_start_time = time.time()
        tfidf_vector = get_tfidf_vector_without_dc()
        classifier = KNeighborsClassifier()

        pipe = Pipeline([('vectorizer', tfidf_vector),
                            ('classifier', classifier)])

        print(f"[kNN] Fitting the model")
        pipe.fit(X_train, y_train)

        print(f"[kNN] Predicting the values")
        y_pred = pipe.predict(X_test)

        print(f"[kNN] Evaluating the prediction")
        metrics = evaluate(y_test, y_pred)
        if bestMetrics is None:
            bestMetrics = metrics
        else:
            if metrics > bestMetrics:
                bestMetrics = metrics
        print(f"[kNN] End of evaluation in {time.time() - config_start_time}")
        self.__bestMetrics = bestMetrics
        print(f"[kNN] Done in {time.time() - start_time}")

