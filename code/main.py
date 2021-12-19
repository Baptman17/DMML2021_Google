import pandas as pd
from LogisticRegression import LogisticRegressionThread
from kNN import KNNThread
from DecisionTree import DecisionTreeThread
from RandomForest import RandomForestThread
from MLP import MLPThread
from mdutils.mdutils import MdUtils
from EvaluationMetrics import EvaluationMetrics
import time
import seaborn as sns
import matplotlib.pyplot as plt
import sys

class ReadmeGenerator:
    def __init__(self, lr0: EvaluationMetrics = None, knn0: EvaluationMetrics = None, dt0: EvaluationMetrics = None, rf0: EvaluationMetrics = None, lr: EvaluationMetrics = None, knn: EvaluationMetrics = None, dt: EvaluationMetrics = None, rf: EvaluationMetrics = None, mlp: EvaluationMetrics = None):
        self.__lr0 = lr0
        self.__knn0 = knn0
        self.__dt0 = dt0
        self.__rf0 = rf0
        self.__lr = lr
        self.__knn = knn
        self.__dt = dt
        self.__rf = rf
        self.__mlp = mlp
        #if running on Mac, change filename to "../Readme"
        self.__mdFile = MdUtils(file_name='README' if sys.platform == "win32" else "../README",title='DMML 2021 Project : Detecting the difficulty level of French texts')

    def generate_readme(self):
        self.__mdFile.new_header(level=1, title="Result without Data Cleaning")
        data = ["/","Logistic regression", "kNN", "Decision Tree", "Random Forests"]
        data.extend(["Precision", self.__lr0.getPrecision(), self.__knn0.getPrecision(),self.__dt0.getPrecision(),self.__rf0.getPrecision()])
        data.extend(["Recall", self.__lr0.getRecall(),self.__knn0.getRecall(),self.__dt0.getRecall(),self.__rf0.getRecall()])
        data.extend(["F1-score", self.__lr0.getF1(),self.__knn0.getF1(),self.__dt0.getF1(),self.__rf0.getF1()])
        data.extend(["Accuracy", self.__lr0.getAccuracy(),self.__knn0.getAccuracy(),self.__dt0.getAccuracy(),self.__rf0.getAccuracy()])
        self.__mdFile.new_table(columns=5, rows=5, text=data)
        self.__mdFile.new_header(level=1, title="Result with Data Cleaning")
        data = ["/","Logistic regression", "kNN", "Decision Tree", "Random Forests", "MLP"]
        data.extend(["Precision", self.__lr.getPrecision(), self.__knn.getPrecision(),self.__dt.getPrecision(),self.__rf.getPrecision(),self.__mlp.getPrecision() if self.__mlp is not None else 0.4827])
        data.extend(["Recall", self.__lr.getRecall(),self.__knn.getRecall(),self.__dt.getRecall(),self.__rf.getRecall(),self.__mlp.getRecall() if self.__mlp is not None else 0.4854])
        data.extend(["F1-score", self.__lr.getF1(),self.__knn.getF1(),self.__dt.getF1(),self.__rf.getF1(),self.__mlp.getF1() if self.__mlp is not None else 0.4837])
        data.extend(["Accuracy", self.__lr.getAccuracy(),self.__knn.getAccuracy(),self.__dt.getAccuracy(),self.__rf.getAccuracy(),self.__mlp.getAccuracy() if self.__mlp is not None else 0.4854])
        self.__mdFile.new_table(columns=6, rows=5, text=data)
        
        self.__mdFile.create_md_file()


if __name__ == '__main__':

    #without data cleaning
    print(f"[Main] Start evaluation without data cleaning")
    start_time = time.time()
    lrThread = LogisticRegressionThread(False)
    lrThread.start()

    kNNThread = KNNThread(False)
    kNNThread.start()

    dtThread = DecisionTreeThread(False)
    dtThread.start()

    rfThread = RandomForestThread(False)
    rfThread.start()

    lr_metrics = lrThread.join()
    kNN_metrics = kNNThread.join()
    dt_metrics = dtThread.join()
    rf_metrics = rfThread.join()

    print(lr_metrics)
    sns.heatmap(pd.DataFrame(lr_metrics.getConfMatrix()), annot=True, cmap='Oranges', fmt='.4g');
    path = "img/LR_without_DC.png" if sys.platform == "win32" else "../img/LR_without_DC.png"
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()
    
    print(kNN_metrics)
    sns.heatmap(pd.DataFrame(kNN_metrics.getConfMatrix()), annot=True, cmap='Oranges', fmt='.4g');
    path = "img/KNN_without_DC.png" if sys.platform == "win32" else "../img/KNN_without_DC.png"
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()

    print(dt_metrics)
    sns.heatmap(pd.DataFrame(dt_metrics.getConfMatrix()), annot=True, cmap='Oranges', fmt='.4g');
    path = "img/DT_without_DC.png" if sys.platform == "win32" else "../img/DT_without_DC.png"
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()

    print(rf_metrics)
    sns.heatmap(pd.DataFrame(rf_metrics.getConfMatrix()), annot=True, cmap='Oranges', fmt='.4g');
    path = "img/RF_without_DC.png" if sys.platform == "win32" else "../img/RF_without_DC.png"
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()

    df = pd.DataFrame({'Accuracy' : [float(lr_metrics.getAccuracy()),float(kNN_metrics.getAccuracy()),float(dt_metrics.getAccuracy()),float(rf_metrics.getAccuracy())]}, index=["LR","kNN","DT","RF"])
    df.plot()
    path = "img/Graph_Acc_without_DC.png" if sys.platform == "win32" else "../img/Graph_Acc_without_DC.png"
    plt.savefig(path)

    print(f"[Main]\t:\tTotal execution time without data cleaning : {time.time() - start_time:.4f} seconds")

    #with data cleaning
    print(f"[Main] Start evaluation with data cleaning")
    start_time = time.time()
    lrThreadDC = LogisticRegressionThread(True)
    lrThreadDC.start()

    kNNThreadDC = KNNThread(True)
    kNNThreadDC.start()

    dtThreadDC = DecisionTreeThread(True)
    dtThreadDC.start()

    rfThreadDC = RandomForestThread(True)
    rfThreadDC.start()

    # It takes about 6 hours to run the MLP classificator
    # Last metrics found were 0.4827, 0.4854, 0.4837 and 0.4854
    # mlpThread = MLPThread()
    # mlpThread.start()

    lr_metricsDC = lrThreadDC.join()
    kNN_metricsDC = kNNThreadDC.join()
    dt_metricsDC = dtThreadDC.join()
    rf_metricsDC = rfThreadDC.join()
    # mlp_metrics = mlpThread.join()

    rg = ReadmeGenerator(lr_metrics, kNN_metrics, dt_metrics, rf_metrics, lr_metricsDC, kNN_metricsDC, dt_metricsDC, rf_metricsDC)
    rg.generate_readme()
    print(f"[Main]\t:\tTotal execution time with data cleaning : {time.time() - start_time:.4f} seconds")



