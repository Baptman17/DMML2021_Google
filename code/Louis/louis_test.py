from util import get_training_data
from util import get_unlabelled_test_data
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import numpy as np
import pandas as pd



from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
df = get_training_data()
df_test = get_unlabelled_test_data()
LR_cv = LogisticRegressionCV(solver='lbfgs', cv=5, max_iter=1000, random_state=72)


tab_final = []
tab_length = []
tab_final_test = []
tab_length_test = []
i = 0

while(i < len(df)):
    cmpt = 0
    txt = df['sentence'][i].split()
    for sw in fr_stop:
        for word in txt:
            if sw == word:
                cmpt = cmpt + 1
    tab_length.append(len(df['sentence'][i]))
    tab_final.append(cmpt / len(df['sentence'][i]))
    i = i+1
df['value'] = tab_final
df['length'] = tab_length
scale_mapper = {"A1":0, "A2":1, "B1":2,"B2":3, "C1":4, "C2":5}
df["y_num"] = df["difficulty"].replace(scale_mapper)
LR_cv.fit(df[['value','length']], df['y_num'])


i = 0

while(i < len(df_test)):
    cmpt_test = 0
    txt_final = df_test['sentence'][i].split()
    for sw in fr_stop:
        for word2 in txt_final:
            if sw == word2:
                cmpt_test = cmpt_test + 1
    tab_length_test.append(len(df_test['sentence'][i]))
    tab_final_test.append(cmpt/len(df_test['sentence'][i]))
    i = i + 1
df_test['value'] = tab_final_test
df_test['length'] = tab_length_test
y_pred = LR_cv.predict(df_test[['value','length']])
df_final = pd.DataFrame()
df_final['id'] = list(range(0, len(df_test)))
scale_mapper = {0:"A1", 1:"A2", 2:"B1",3:"B2", 4:"C1", 5:"C2"}
df_final['difficulty'] = y_pred
df_final['difficulty'] = df_final['difficulty'].replace(scale_mapper)
print(df_final)
df_final.to_csv("submission_test.csv", index = False)


