import config

import pandas as pd
import numpy as np
from sklearn import linear_model,metrics
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

df=pd.read_csv(config.FOLDED_DATA)

for fold in range(4):
    test_data=df[df['fold']==fold]
    training_data=df[df['fold']!=fold]

    count_vec=CountVectorizer(tokenizer=word_tokenize,token_pattern=None)

    count_vec.fit(training_data['Title'])

    X_train=count_vec.transform(training_data['Title'])
    X_test=count_vec.transform(test_data['Title'])

    