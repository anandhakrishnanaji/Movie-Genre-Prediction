import config
import pandas as pd
from sklearn.model_selection import StratifiedKFold


df=pd.read_csv(config.TRAINING_FILE)

df=df.sample(frac=1)
df["fold"]=-1

spl=StratifiedKFold(n_splits=4)

for fold,(t_,v_) in enumerate(spl.split(X=df,y=df['Drama'])):
    df.loc[v_,"fold"]=fold

df.to_csv('../input/folded_data.csv',index=False)
