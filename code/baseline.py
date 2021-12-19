from util import get_training_data
df = get_training_data()
valcount = df['difficulty'].value_counts()
print("Baseline : " + str(max(valcount) / len(df['difficulty'])))
#Baseline : 0.169375