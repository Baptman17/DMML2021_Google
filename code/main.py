from util import get_training_data
from LogisticRegression import LogisticRegressionThread
from kNN import kNNThread
from DecisionTree import DecisionTreeThread
from RandomForest import RandomForestThread
from MLP import MLPThread
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from mdutils.mdutils import MdUtils
import threading
from EvaluationMetrics import EvaluationMetrics
import time

class ReadmeGenerator:
    def __init__(self, lr: EvaluationMetrics, knn: EvaluationMetrics, dt: EvaluationMetrics, rf: EvaluationMetrics):
        self.__lr = lr
        self.__knn = knn
        self.__dt = dt
        self.__rf = rf
        self.__mdFile = MdUtils(file_name="../README")

    def generate_redame(self):
        self.__mdFile.new_header(level=1, title="UNIL_DataMining_TextAnalytics")
        data = ["/","Logistic regression", "kNN", "Decision Tree", "Random Forests", "MLP"]
        data.extend(["Precision", self.__lr.getPrecision(), self.__knn.getPrecision(),self.__dt.getPrecision(),self.__rf.getPrecision(),""])
        data.extend(["Recall", self.__lr.getRecall(),self.__knn.getRecall(),self.__dt.getRecall(),self.__rf.getRecall(),""])
        data.extend(["F1-score", self.__lr.getF1(),self.__knn.getF1(),self.__dt.getF1(),self.__rf.getF1(),""])
        data.extend(["Accuracy", self.__lr.getAccuracy(),self.__knn.getAccuracy(),self.__dt.getAccuracy(),self.__rf.getAccuracy(),""])
        self.__mdFile.new_table(columns=6, rows=5, text=data)
        self.__mdFile.create_md_file()


if __name__ == '__main__':
    start_time = time.time()
    lrThread = LogisticRegressionThread()
    lrThread.start()

    kNNThread = kNNThread()
    kNNThread.start()

    dtThread = DecisionTreeThread()
    dtThread.start()

    rfThread = RandomForestThread()
    rfThread.start()

    #mlpThread = MLPThread()
    #mlpThread.start()

    lr_metrics = lrThread.join()
    kNN_metrics = kNNThread.join()
    dt_metrics = dtThread.join()
    rf_metrics = rfThread.join()
    #mlpMetrics = mlpThread.join()

    rg = ReadmeGenerator(lr_metrics, kNN_metrics, dt_metrics, rf_metrics)
    rg.generate_redame()
    print(f"[Main]\t:\tTotal execution time : {time.time() - start_time}")



