from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score


class EvaluationMetrics:
    def __init__(self, true,pred):
        self.__precision = round(precision_score(true, pred, average='weighted'),4),
        self.__recall = round(recall_score(true, pred, average='weighted'),4),
        self.__f1 = round(f1_score(true, pred, average='weighted'),4),
        self.__accuracy = round(accuracy_score(true, pred),4),
        self.__confusion_matrix = confusion_matrix(true,pred),
        self.__config = None

    def getPrecision(self):
        return self.__precision[0]

    def getRecall(self):
        return self.__recall[0]

    def getF1(self):
        return self.__f1[0]

    def getAccuracy(self):
        return self.__accuracy[0]
    
    def getConfMatrix(self):
        return self.__confusion_matrix[0]

    def getSubFile(self):
        df_final = pd.DataFrame()
        df_final['id'] = list(range(0, len(df_test)))
        df_final['difficulty'] = y_pred_test
        df_final.to_csv("submission_LRpv.csv", index = False)

    def setConfig(self, config):
        self.__config = config

    def __gt__(self, other):
        return self.__accuracy > other.getAccuracy()

    def __str__(self):
        txtstr = (f"Precision\t{self.__precision}"
              f"\nRecall\t{self.__recall}"
              f"\nF1-score\t{self.__f1}"
              f"\nAccuracy\t{self.__accuracy}"
              f"\nConfusion Matrix\n{self.__confusion_matrix}")
        if self.__config is not None:
            txtstr = txtstr + f"\nConfig\t{self.__config}"
        return txtstr 
