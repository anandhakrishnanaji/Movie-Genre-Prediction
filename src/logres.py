import config
import pipeline_gen

import pandas as pd

from sklearn import linear_model,metrics,multiclass

def get_accuracies(pred,true):
    measures={}
    measures['Accuracy']=metrics.accuracy_score(pred,true)

    measures['Micro Precision']=metrics.precision_score(pred,true,average="micro")
    measures['Macro Precision']=metrics.precision_score(pred,true,average="macro")

    measures['Micro Recall']=metrics.recall_score(pred,true,average="micro")
    measures['Macro Recall']=metrics.recall_score(pred,true,average="macro")

    measures['Micro f1']=metrics.f1_score(pred,true,average="micro")
    measures['Macro f1']=metrics.f1_score(pred,true,average="macro")

    return measures

def print_measures(measures):
    for key in measures:
        print('{} : {}'.format(key,measures[key]))


df=pd.read_csv(config.FOLDED_DATA)

for fold in range(4):
    print('FOLD {}'.format(fold))

    test_data=df[df['fold']==fold]
    train_data=df[df['fold']!=fold]

    X_train=train_data['Title']
    X_test=test_data['Title']

    y_train=train_data.iloc[:,1:]
    y_test=test_data.iloc[:,1:]

    model=multiclass.OneVsRestClassifier(linear_model.LogisticRegression(random_state=1),n_jobs=-1)

    print('MODEL DEFINED...')

    pipeline=pipeline_gen.create_pipeline(model)

    print('FITTING PIPELINE...')
    pipeline.fit(X_train,X_test)

    print('PREDICTING VALUES...')
    y_pred_train=pipeline.predict(X_train)
    y_pred_test=pipeline.predict(X_test)

    print('FETCHING ACCURACIES...')
    measures_train=get_accuracies(y_pred_train,y_train)
    measures_test=get_accuracies(y_pred_test,y_test)

    print('TRAIN SCORE (FOLD {})'.format(fold),end='\n\n')
    print_measures(measures_train)

    print('TEST SCORE (FOLD {})'.format(fold),end='\n\n')
    print_measures(measures_test)

