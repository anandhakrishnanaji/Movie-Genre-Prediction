import config
import pipeline_gen

import os
import pandas as pd
import joblib
from sklearn import linear_model,metrics,multiclass,model_selection

THRESHOLD=0.4

df=pd.read_csv(config.FOLDED_DATA)

X=df['Title']
y=df.iloc[:,1:-1]

# model=multiclass.OneVsRestClassifier(linear_model.LogisticRegression(random_state=1,n_jobs=-1,solver='liblinear'),n_jobs=-1)
# pipeline=pipeline_gen.create_pipeline(model)

# param_grid={
#     'stopwords__remove':[True,False],
#     'stem__lemmatize':[True,False],
#     'embed__option':['tfidf','countvec'],
#     'embed__ngram_range':[(1,1),(1,2),(1,3)],
#     'model__estimator__max_iter':[2000,3000],
    
# }

# grid=model_selection.GridSearchCV(pipeline,param_grid=param_grid,scoring='f1_micro',cv=4,n_jobs=-1,return_train_score=True,verbose=1)

# grid.fit(X,y)

# print(grid.best_params_)
# print(grid.best_estimator_)
# print(grid.best_score_)

model_path=os.path.join(config.MODEL_OUTPUT,"logres_best.joblib")

# joblib.dump(grid.best_estimator_,model_path)

model=joblib.load(model_path)

y_pred=model.predict_proba(X)
y_pred[y_pred<0.4]=0
y_pred[y_pred>=0.4]=1

print(metrics.f1_score(y,y_pred,average="micro"))