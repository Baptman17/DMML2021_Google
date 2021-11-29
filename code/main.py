from util import get_training_data
import spacy
from spacy import displacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df = get_training_data()
print(df.head(10))