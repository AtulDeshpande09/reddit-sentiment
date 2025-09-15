import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd

data = pd.read_csv('data/Reddit_Data.csv')
df = pd.DataFrame(data)
df = df.dropna()

def process(comment):
    comment = re.sub(r'[^a-zA-Z]', ' ' , comment)
    comment = comment.lower()
    comment = comment.split()
    stemmer = PorterStemmer()
    comment = [ stemmer.stem(word) for word in comment if word not in set(stopwords.words("english"))]
    return " ".join(comment)


# preprocessing the comments
df['processed'] = df["clean_comment"].apply(process)

df.to_csv("data/preprocessed.csv",index=False)