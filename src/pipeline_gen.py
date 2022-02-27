import re
import string

import config

import pandas as pd

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,ENGLISH_STOP_WORDS
from sklearn.base import BaseEstimator,TransformerMixin

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

from gensim.models import word2vec,fasttext

def sentence2vec(s,embedding_dict):
    words=word_tokenize(s)
    words=[w for w in words if w.isalpha()]

    M=[]

    for w in words:
        if w in embedding_dict:
            M.append(embedding_dict[w])

    if len(M)==0:
        return np.zeros(300)

    M=np.array(M)

    v= M.sum(axis=0)

    return v/np.sqrt((v**2).sum())

class CleanData(BaseEstimator,TransformerMixin):
    def _clean(self,s):
        return re.sub(f'[{re.escape(string.punctuation)}|{re.escape(string.digits)}]',"",s.lower()).strip()

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X=X.apply(lambda s:self._clean(s))
        return X

class StemLemmatizeData(BaseEstimator,TransformerMixin):
    def __init__(self,lemmatize=False):
        self._lemmatize=lemmatize
        self._stemmer=SnowballStemmer("english")
        self._lemmatizer=WordNetLemmatizer()

    def _stem(self,s):
        return [self._stemmer.stem(x) for x in s.split()].join(' ')
    
    def _lemma(self,s):
        return (' ').join([self._lemmatizer.lemmatize(x) for x in s.split()])
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self._lemmatize:
            return X.apply(lambda s: self._lemma(s))
        else:
            return X.apply(lambda s: self._stem(s))

class StopWordsRemoval(BaseEstimator,TransformerMixin):
    def _removeStopWords(self,s):
        tokens=word_tokenize(s)
        filtered_sentence = [w for w in tokens if not w.lower() in self._swords]
        return ' '.join(filtered_sentence)

    def __init__(self):
        self._swords=stopwords.words('english')

    def fit(self,X,y):
        return self
    
    def transform(self,X,y):
        return X.apply(lambda s: self._removeStopWords(s))


vectorize={'tfidf':TfidfVectorizer(),'countvec':CountVectorizer()}


def create_pipeline(*args,**kwargs):
    pass

print(len(ENGLISH_STOP_WORDS))
# print(len(stopwords.words("english")))