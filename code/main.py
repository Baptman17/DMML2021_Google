from util import get_training_data
from LogisticRegression import get_lr_metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from mdutils.mdutils import MdUtils

df = get_training_data()

# What we should do (according to the guidelines)

# Logistic Regression (without data cleaning)
# Train a simple logistic regression model using a Tfidf vectoriser.
# TODO
# Calculate accuracy, precision, recall and F1 score on the test set.
# TODO
# Have a look at the confusion matrix and identify a few examples of sentences that are not well classified.
# TODO
# Generate your first predictions on the unlabelled_test_data.csv. make sure your predictions match the format of the
# unlabelled_test_data.csv.

# KNN (without data cleaning)
# Train a KNN classification model using a Tfidf vectoriser. Show the accuracy, precision, recall and F1 score on the
# test set.
# TODO
# Try to improve it by tuning the hyper parameters (n_neighbors, p, weights).
# TODO

# Decision Tree Classifier (without data cleaning)
# Train a Decison Tree classifier, using a Tfidf vectoriser. Show the accuracy, precision, recall and F1 score on the
# test set.
# TODO
# Try to improve it by tuning the hyper parameters (max_depth, the depth of the decision tree).
# TODO

# Random Forest Classifier (without data cleaning)
# Try a Random Forest Classifier, using a Tfidf vectoriser. Show the accuracy, precision, recall and F1 score on the
# test set.
# TODO

# Any other technique, including data cleaning if necessary
# Try to improve accuracy by training a better model using the techniques seen in class, or combinations of them.
# As usual, show the accuracy, precision, recall and f1 score on the test set.
# TODO
class ReadmeGenerator:
    def __init__(self, lr):
        self.__lr = lr
        self.__mdFile = MdUtils(file_name="../README")

    def generate_redame(self):
        self.__mdFile.new_header(level=1, title="UNIL_DataMining_TextAnalytics")
        data = ["","Logistic regression", "kNN Decision Tree", "Random Forests", "Paragraph", "Any other technique"]
        data.extend(["Precision", self.__lr.getPrecision(),"","","",""])
        data.extend(["Recall", self.__lr.getRecall(),"","","",""])
        data.extend(["F1-score", self.__lr.getF1(),"","","",""])
        data.extend(["Accuracy", self.__lr.getAccuracy(),"","","",""])
        self.__mdFile.new_table(columns=6, rows=5, text=data)
        self.__mdFile.create_md_file()

if __name__ == '__main__':
    print("Logistic regression")
    lrMetrics = get_lr_metrics()

    rg = ReadmeGenerator(lrMetrics)
    rg.generate_redame()



