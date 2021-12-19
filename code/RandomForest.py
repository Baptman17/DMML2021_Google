from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import util
from util import  get_training_data, get_tfidf_vector, evaluate
import threading
import time

class RandomForestThread(threading.Thread):
    def __init__(self, with_dc: bool):
        threading.Thread.__init__(self)
        self.__bestMetrics = None
        self.__with_dc = with_dc

    def run(self):
        if self.__with_dc:
            self.get_metrics_with_dc()
        else:
            self.get_metrics_without_dc()

    def join(self):
        threading.Thread.join(self)
        return self.__bestMetrics

    def get_metrics_with_dc(self):
        print("[RF]\t:\tGetting data")
        start_time = time.time()
        df = get_training_data()
        df["difficulty"] = df["difficulty"].astype("category")

        print("[RF]\t:\tSplitting data")
        X = df['sentence']
        y = df['difficulty']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        bestMetrics = None
        configs = util.configs()
        for index, config in enumerate(configs):
            config_start_time = time.time()
            configId = f"config {index + 1}/{len(configs)}"
            print(f"[RF]\t:\t Starting with {configId}")
            tfidf_vector = get_tfidf_vector(config=config, with_dc=True)
            #classifier = RandomForestClassifier(n_estimators=2000)
            classifier = RandomForestClassifier()

            pipe = Pipeline([('vectorizer', tfidf_vector),
                             ('classifier', classifier)])

            print(f"[RF] ({configId})\t:\tFitting the model")
            pipe.fit(X_train, y_train)

            print(f"[RF] ({configId})\t:\tPredicting the values")
            y_pred = pipe.predict(X_test)

            print(f"[RF] ({configId})\t:\tEvaluating the prediction")
            metrics = evaluate(y_test, y_pred)
            metrics.setConfig(config)
            if bestMetrics is None:
                bestMetrics = metrics
            else:
                if metrics > bestMetrics:
                    bestMetrics = metrics
            print(f"[RF] ({configId})\t:\tEnd of evaluation in {time.time() - config_start_time:.4f} seconds")
        self.__bestMetrics = bestMetrics
        print(f"[RF]\t:\tDone in {time.time() - start_time:.4f} seconds")

    def get_metrics_without_dc(self):
        print("[RF] Getting data")
        start_time = time.time()
        df = get_training_data()
        df["difficulty"] = df["difficulty"].astype("category")

        print("[RF] Splitting data")
        X = df['sentence']
        y = df['difficulty']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        bestMetrics = None
        config_start_time = time.time()
        tfidf_vector = get_tfidf_vector()
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
        print(f"[RF] End of evaluation without data cleaning in {time.time() - config_start_time:.4f} seconds")
        self.__bestMetrics = bestMetrics
