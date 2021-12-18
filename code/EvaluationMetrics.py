class EvaluationMetrics:
    def __init__(self, precision, recall, f1, accuracy):
        self.__precision = round(precision,4)
        self.__recall = round(recall,4),
        self.__f1 = round(f1,4)
        self.__accuracy = round(accuracy,4)
        self.__config = None

    def getPrecision(self):
        return self.__precision

    def getRecall(self):
        return self.__recall

    def getF1(self):
        return self.__f1

    def getAccuracy(self):
        return self.__accuracy

    def setConfig(self, config):
        self.__config = config

    def __gt__(self, other):
        return self.__accuracy > other.getAccuracy()

    def __str__(self):
        return(f"Precision\t{self.__precision}"
              f"\nRecall\t{self.__recall}"
              f"\nF1-score\t{self.__f1}"
              f"\nAccuracy\t{self.__accuracy}"
              f"\nConfig\t{self.__config}")
