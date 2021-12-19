from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from util import get_tfidf_vector_without_dc
import util
from util import get_training_data, get_tfidf_vector, evaluate, configs
import pandas as pd
import threading
import time

class DecisionTreeThread_without_dc(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.__bestMetrics = None

    def run(self):
        self.get_metrics()

    def join(self):
        threading.Thread.join(self)
        return self.__bestMetrics

    def get_metrics(self):
        print("[DT] Getting data")
        start_time = time.time()
        df = get_training_data()
        df["difficulty"] = df["difficulty"].astype("category")

        print("[DT] Splitting data")
        X = df['sentence']
        y = df['difficulty']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        bestMetrics = None
        config_start_time = time.time()
        tfidf_vector = get_tfidf_vector_without_dc()
        classifier = DecisionTreeClassifier()

        pipe = Pipeline([('vectorizer', tfidf_vector),
                        ('classifier', classifier)])

        print(f"[DT] Fitting the model")
        pipe.fit(X_train, y_train)

        print(f"[DT] Predicting the values")
        y_pred = pipe.predict(X_test)

        print(f"[DT] Evaluating the prediction")
        metrics = evaluate(y_test, y_pred)
        if bestMetrics is None:
            bestMetrics = metrics
        else:
            if metrics > bestMetrics:
                bestMetrics = metrics
        print(f"[DT] End of evaluation in {time.time() - config_start_time}")
        self.__bestMetrics = bestMetrics
        print(f"[DT] Done in {time.time() - start_time}")

