import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

dataset_df = pd.read_csv(r"C:\Users\ghibl\ICR\data\input\train.csv")
dataset_df.drop("Id",axis = 1, inplace = True)

le = LabelEncoder()
dataset_df["EJ"] = le.fit_transform(dataset_df["EJ"])

X_df = dataset_df.drop("Class",axis =1)
y_df = dataset_df["Class"].copy()
skf = StratifiedKFold(n_splits = 4, shuffle = True, random_state = 42)
scores = []

for tr_idx, va_idx in skf.split(X_df, y_df):
    X_train, X_valid = X_df.iloc[tr_idx], X_df.iloc[va_idx]
    y_train, y_valid = y_df.iloc[tr_idx], y_df.iloc[va_idx]

    dtrain = xgb.DMatrix(X_train, label = y_train)
    dvalid = xgb.DMatrix(X_valid, label = y_valid)

    params = {'objective': 'binary:logistic','silent': 1, 'random_state': 42
              , 'learning_rate' : 0.1}
    num_round = 500

    watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
    model = xgb.train(params, dtrain, num_round, 
                      evals = watchlist, early_stopping_rounds = 20)

    y_pred = model.predict(dvalid)
    score = log_loss(y_valid, y_pred)
    scores.append(score)
    print(f'logloss: {score:.4f}')

score_CV = np.mean(scores)
print(score_CV)

