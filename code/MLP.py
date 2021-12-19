from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import util
from util import get_training_data, get_tfidf_vector, evaluate
from sklearn.metrics import accuracy_score
import pandas as pd
import threading
import time


class MLPThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.__bestMetrics = None

    def run(self):
        self.get_metrics()

    def join(self):
        threading.Thread.join(self)
        return self.__bestMetrics

    def get_metrics(self):
        print("[MLP]\t:\tGetting data")
        start_time = time.time()
        df = get_training_data()
        df["difficulty"] = df["difficulty"].astype("category")

        print("[MLP]\t:\tSplitting data")
        X = df['sentence']
        y = df['difficulty']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        bestMetrics = None
        configs = util.configs()
        for index, config in enumerate(configs):
            config_start_time = time.time()
            configId = f"config {index + 1}/{len(configs)}"
            print(f"[MLP]\t:\t Starting with {configId}")
            tfidf_vector = get_tfidf_vector(config=config, with_dc=True)
            classifier = MLPClassifier(max_iter=10000,
                                       activation="tanh",
                                       solver="sgd")

            pipe = Pipeline([('vectorizer', tfidf_vector),
                             ('classifier', classifier)])

            print(f"[MLP] ({configId})\t:\tFitting the model")
            pipe.fit(X_train, y_train)

            print(f"[MLP] ({configId})\t:\tPredicting the values")
            y_pred = pipe.predict(X_test)

            print(f"[MLP] ({configId})\t:\tEvaluating the prediction")
            metrics = evaluate(y_test, y_pred)
            metrics.setConfig(config)
            if bestMetrics is None:
                bestMetrics = metrics
            else:
                if metrics > bestMetrics:
                    bestMetrics = metrics
            print(f"[MLP] ({configId})\t:\tEnd of evaluation in {time.time() - config_start_time}")
        self.__bestMetrics = bestMetrics
        print(f"[MLP]\t:\tDone in {time.time() - start_time}")
