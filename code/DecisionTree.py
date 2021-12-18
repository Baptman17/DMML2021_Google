from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import util
from util import get_training_data, get_tfidf_vector, evaluate, configs
import pandas as pd
import threading
import time

class DecisionTreeThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.__bestMetrics = None

    def run(self):
        self.get_metrics()

    def join(self):
        threading.Thread.join(self)
        return self.__bestMetrics

    def get_metrics(self):
        print("[DT]\t:\tGetting data")
        start_time = time.time()
        df = get_training_data()
        df["difficulty"] = df["difficulty"].astype("category")

        print("[DT]\t:\tSplitting data")
        X = df['sentence']
        y = df['difficulty']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

        bestMetrics = None
        configs = util.configs()
        for index, config in enumerate(configs):
            config_start_time = time.time()
            configId = f"config {index + 1}/{len(configs)}"
            print(f"[DT]\t:\t Starting with {configId}")
            tfidf_vector = get_tfidf_vector(config)
            classifier = DecisionTreeClassifier()

            pipe = Pipeline([('vectorizer', tfidf_vector),
                            ('classifier', classifier)])

            print(f"[DT] ({configId})\t:\tFitting the model")
            pipe.fit(X_train, y_train)

            print(f"[DT] ({configId})\t:\tPredicting the values")
            y_pred = pipe.predict(X_test)

            print(f"[DT] ({configId})\t:\tEvaluating the prediction")
            metrics = evaluate(y_test, y_pred)
            metrics.setConfig(config)
            if bestMetrics is None:
                bestMetrics = metrics
            else:
                if metrics > bestMetrics:
                    bestMetrics = metrics
            print(f"[DT] ({configId})\t:\tEnd of evaluation in {time.time() - config_start_time}")
        self.__bestMetrics = bestMetrics
        print(f"[DT]\t:\tDone in {time.time() - start_time}")

