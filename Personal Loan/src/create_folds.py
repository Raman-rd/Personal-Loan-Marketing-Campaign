import pandas as pd
import numpy as np
from sklearn import model_selection


if __name__=="__main__":

    df = pd.read_csv("input/cleaned_data.csv")

    df["kfold"] = -1

    df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5)

    y = df["Personal Loan"].values

    for f,(t,v) in enumerate(kf.split(X=df,y=y)):

        df.loc[v,"kfold"] = f

    df.to_csv("input/data_folds.csv",index=False)