import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import util
from util import get_training_data, get_tfidf_vector, evaluate, get_unlabelled_test_data
import threading
import time
import sys

class LogisticRegressionThread(threading.Thread):
    def __init__(self, with_dc : bool):
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
        print("[LR]\t:\tGetting data")
        start_time = time.time()
        df = get_training_data()
        df["difficulty"] = df["difficulty"].astype("category")

        print("[LR]\t:\tSplitting data")
        X = df['sentence']
        y = df['difficulty']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

        bestMetrics = None
        bestModel = None
        configs = util.configs()
        for index, config in enumerate(configs):
            config_start_time = time.time()
            configId = f"config {index+1}/{len(configs)}"
            print(f"[LR]\t:\t Starting with {configId}")
            tfidf_vector = get_tfidf_vector(config=config, with_dc=True)
            classifier = LogisticRegression(max_iter=150000, solver = 'saga', random_state=0)

            pipe = Pipeline([('vectorizer', tfidf_vector),
                             ('classifier', classifier)])

            print(f"[LR] ({configId})\t:\tFitting the model")
            pipe.fit(X_train, y_train)

            print(f"[LR] ({configId})\t:\tPredicting the values")
            y_pred = pipe.predict(X_test)

            print(f"[LR] ({configId})\t:\tEvaluating the prediction")
            metrics = evaluate(y_test, y_pred)
            metrics.setConfig(config)
            if bestMetrics is None:
                bestMetrics = metrics
            else:
                if metrics > bestMetrics:
                    bestModel = pipe
                    bestMetrics = metrics
            print(f"[LR] ({configId})\t:\tEnd of evaluation in {time.time() - config_start_time:.4f} seconds")
        df_final = pd.DataFrame()
        df_test = get_unlabelled_test_data()
        df_final['id'] = list(range(0, len(df_test)))
        df_final['difficulty'] = bestModel.predict(df_test['sentence'])
        path = "submissions/submission_LR_with_DC.csv" if sys.platform == "win32" else "../submissions/sumbission_LR_with_DC.csv"
        df_final.to_csv(path, index = False)
        self.__bestMetrics = bestMetrics
        
        print(f"[LR]\t:\tDone in {time.time() - start_time:.4f} seconds")

    def get_metrics_without_dc(self):
        print("[LR] Getting data")
        start_time = time.time()
        df = get_training_data()
        df["difficulty"] = df["difficulty"].astype("category")

        print("[LR] Splitting data")
        X = df['sentence']
        y = df['difficulty']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        config_start_time = time.time()
        tfidf_vector = get_tfidf_vector()
        classifier = LogisticRegression()

        pipe = Pipeline([('vectorizer', tfidf_vector),
                        ('classifier', classifier)])

        print(f"[LR] Fitting the model")
        pipe.fit(X_train, y_train)

        print(f"[LR] Predicting the values")
        y_pred = pipe.predict(X_test)

        print(f"[LR] Evaluating the prediction")
        metrics = evaluate(y_test, y_pred)
        print(f"[LR] End of evaluation without data cleaning in {time.time() - config_start_time:.4f} seconds")
        df_final = pd.DataFrame()
        df_test = get_unlabelled_test_data()
        df_final['id'] = list(range(0, len(df_test)))
        df_final['difficulty'] = pipe.predict(df_test['sentence'])
        path = "submissions/submission_LR_without_DC.csv" if sys.platform == "win32" else "../submissions/submission_LR_without_DC.csv"
        df_final.to_csv(path, index = False)
        self.__bestMetrics = metrics
