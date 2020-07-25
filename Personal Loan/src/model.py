import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
import joblib


def run(fold):

    df = pd.read_csv("input/data_folds.csv")

    df_train = df[df.kfold != fold].reset_index(drop = True)

    df_valid = df[df.kfold == fold].reset_index(drop = True)

    xtrain = df_train.drop("Personal Loan",axis=1).values
    ytrain = df_train["Personal Loan"].values

    xvalid = df_valid.drop("Personal Loan",axis=1).values
    yvalid = df_valid["Personal Loan"].values

    scaler = preprocessing.StandardScaler()

    xtrain = scaler.fit_transform(xtrain)

    xvalid = scaler.transform(xvalid)

    model = linear_model.LogisticRegression()

    model.fit(xtrain,ytrain)

    yhat = model.predict(xvalid)

    roc = metrics.roc_auc_score(yhat,yvalid)
    acc = metrics.accuracy_score(yhat,yvalid)
    prec = metrics.precision_score(yhat,yvalid)
    recall = metrics.recall_score(yhat,yvalid)
    print(f"ROC {roc} Accuracy {acc} fold {fold} precision {prec} recall {recall}")
    
    
    joblib.dump("model/",f"{fold}dt.bin")

if __name__ =="__main__":

    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)