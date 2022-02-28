import re
import io
import string

import config

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import BaseEstimator,TransformerMixin

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

# from gensim.models import word2vec,fasttext

class CleanData(BaseEstimator,TransformerMixin):
    def _clean(self,s):
        return re.sub(f'[{re.escape(string.punctuation)}|{re.escape(string.digits)}]',"",s.lower()).strip()

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X=X.apply(lambda s:self._clean(s))
        print('DATA CLEANED...')
        return X

class StemLemmatizeData(BaseEstimator,TransformerMixin):
    def __init__(self,lemmatize=False):
        self._lemmatize=lemmatize
        self._stemmer=SnowballStemmer("english")
        self._lemmatizer=WordNetLemmatizer()

    def _stem(self,s):
        return ' '.join([self._stemmer.stem(x) for x in s.split()])
    
    def _lemma(self,s):
        return ' '.join([self._lemmatizer.lemmatize(x) for x in s.split()])
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self._lemmatize:
            X=X.apply(lambda s: self._lemma(s))
        else:
            X=X.apply(lambda s: self._stem(s))
        print('DATA STEMMED/LEMMATIZED...')
        return X

    def get_params(self,deep=False):
        return {'lemmatize':self._lemmatize}
    
    def set_params(self,**parameters):
        self._lemmatize=parameters['lemmatize']
        return self

class StopWordsRemoval(BaseEstimator,TransformerMixin):
    def _removeStopWords(self,s):
        tokens=word_tokenize(s)
        filtered_sentence = [w for w in tokens if not w.lower() in self._swords]
        return ' '.join(filtered_sentence)

    def __init__(self,remove=False):
        self._swords=stopwords.words('english')
        self._remove=remove

    def fit(self,X,y):
        return self
    
    def transform(self,X):
        print('REMOVING STOPWORDS...')
        if self._remove:
            return X.apply(lambda s: self._removeStopWords(s))
        return X

    def get_params(self,deep=False):
        return {"remove":self._remove}
    
    def set_params(self,**parameters):
        self._remove=parameters["remove"]
        return self

class FastTextVectorizer(BaseEstimator,TransformerMixin):
    def _sentence2vec(self,s):
        words=word_tokenize(s)
        words=[w for w in words if w.isalpha()]

        M=[]

        for w in words:
            if w in self.embedding_dict:
                M.append(self.embedding_dict[w])

        if len(M)==0:
            return np.zeros(300)

        M=np.array(M)

        v= M.sum(axis=0)

        return v/np.sqrt((v**2).sum())

    def _load_vector(self):
        fin=io.open(config.VECTOR_FILE,'r',encoding='utf-8',newline='\n',errors='ignore')
        n,d=map(int,fin.readline().split())
        data={}
        for line in fin:
            tokens=line.rstrip().split(' ')
            data[tokens[0]]=list(map(float,tokens[1:]))
        return data

    def __init__(self):
        self.embedding_dict=self._load_vector()
        print('VECTOR LOADED...')
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X=X.apply(lambda s: self._sentence2vec(s))
        print('WORD EMBEDDINGS APPLIED...')
        return X

class Vectorizer(BaseEstimator,TransformerMixin):
    def __init__(self,option='countvec',ngram_range=(1,1)):
        self._option=option
        self._ngram_range=ngram_range
    
    def fit(self,X,y=None):
        # print(self._option)
        if(self._option=='countvec'):
            self._model=CountVectorizer(ngram_range=self._ngram_range,tokenizer=word_tokenize,token_pattern=None).fit(X)
        elif(self._option=='tfidf'):
            self._model=TfidfVectorizer(ngram_range=self._ngram_range,tokenizer=word_tokenize,token_pattern=None).fit(X)
        else:
            self._model=FastTextVectorizer().fit(X)
        return self
    
    def transform(self,X):
        print("TRANSFORMING TO VECTOR")
        return self._model.transform(X)
    
    def get_params(self,deep=False):
        return {'option':self._option,'ngram_range':self._ngram_range}
    
    def set_params(self,**parameters):
        self._option=parameters['option']
        self._ngram_range=parameters['ngram_range']
        return self
        

def create_pipeline(model,stopwords=False,lemmatize=False,embedding='countvec'):

    print('CREATING PIPELINE...')

    transformers=[('clean',CleanData()),
                ('stopwords',StopWordsRemoval(remove=stopwords)),
                ('stem',StemLemmatizeData(lemmatize=lemmatize)),
                ('embed',Vectorizer(option=embedding)),
                ('model',model)]

    return Pipeline(transformers)
