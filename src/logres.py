from locale import normalize

import config
import pipeline_gen

import pandas as pd

import numpy as np

from sklearn import linear_model,metrics,multiclass,naive_bayes

measures_array=[]

# def aggregate_scores():
#     aggr=measures_array[0]
#     for dic in range(1,4):
#         for key in measures_array[dic]:
#             aggr[key]+=measures_array[dic][key]
#     for key in aggr:
#         aggr[key]/=4
#     return aggr

def get_accuracies(pred,true):
    measures=dict()

    measures['Micro Precision']=metrics.precision_score(true,pred,average="micro")
    measures['Macro Precision']=metrics.precision_score(true,pred,average="macro")

    measures['Micro Recall']=metrics.recall_score(true,pred,average="micro")
    measures['Macro Recall']=metrics.recall_score(true,pred,average="macro")

    measures['Micro f1']=metrics.f1_score(true,pred,average="micro")
    measures['Macro f1']=metrics.f1_score(true,pred,average="macro")

    measures['Hamming Loss']=metrics.hamming_loss(true,pred)
    measures['Zero One Loss']=metrics.zero_one_loss(true,pred)


    measures_array.append(measures)
    return measures

def print_measures(measures):
    for key in measures:
        print('{} : {}'.format(key,measures[key]))


df=pd.read_csv(config.FOLDED_DATA)

genres=np.array(df.columns[1:-1])

for fold in range(4):
    print('\nFOLD {}\n'.format(fold))

    test_data=df[df['fold']==fold]
    train_data=df[df['fold']!=fold]

    X_train=train_data['Title']
    X_test=test_data['Title']

    y_train=train_data.iloc[:,1:-1]
    y_test=test_data.iloc[:,1:-1]

    model=multiclass.OneVsRestClassifier(linear_model.LogisticRegression(random_state=1,n_jobs=-1,max_iter=1000),n_jobs=-1)

    print('MODEL DEFINED...')

    pipeline=pipeline_gen.create_pipeline(model)

    print('FITTING PIPELINE...')
    pipeline.fit(X_train,y_train)


    print('PREDICTING VALUES...')
    y_pred_train=pipeline.predict_proba(X_train)
    y_pred_test=pipeline.predict_proba(X_test)

    y_pred_train[y_pred_train<0.4]=0
    y_pred_train[y_pred_train>=0.4]=1

    y_pred_test[y_pred_test<0.4]=0
    y_pred_test[y_pred_test>=0.4]=1

    ##
    # thresh=numpy.linspace(0.3,0.8,11)
    # y_pp_train=pipeline.predict_proba(X_train)
    # for i in thresh:
    #     ypp=y_pp_train.copy()
    #     ypp[y_pp_train<i]=0
    #     ypp[y_pp_train>=i]=1
    #     mt=get_accuracies(ypp,y_train)
    #     print("\n{}\n\n".format(i))
    #     print_measures(mt)
    # input()
    ##


    print('FETCHING ACCURACIES...')
    measures_train=get_accuracies(y_pred_train,y_train)
    measures_test=get_accuracies(y_pred_test,y_test)

    # print('TRAIN SCORE (FOLD {})'.format(fold),end='\n\n')
    # print_measures(measures_train)

    # print('TEST SCORE (FOLD {})'.format(fold),end='\n\n')
    # print_measures(measures_test)

    movie_name='Happy Husbands'
    pred_proba=pipeline.predict_proba(pd.Series([movie_name,])).flatten()
    predicted_genres=genres[pred_proba>0.4]

    print(movie_name,predicted_genres)


# print('MEAN SCORES')
# print_measures(aggregate_scores())

