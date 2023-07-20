import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

'''
def balanced_logloss(y_pred,y_true):
    nc = np.bincount(y_true)
    return log_loss(y_true, y_pred, sample_weight = 1/nc[y_true], eps=1e-15)
'''

def balanced_logloss(y_pred,y_true):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(float)
    
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    individual_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    class_weights = np.where(y_true == 1, 1/np.sum(y_true == 1) , 1/np.sum(y_true == 0))
    weighted_loss = individual_loss * class_weights
    
    balanced_logloss_value = np.sum(weighted_loss)/2
    return balanced_logloss_value


def original_balanced_logloss_metric(y_pred, trn_data):
    y_true = trn_data.get_label()
    pred = 1/(1+np.exp(-y_pred))
    return 'balanced_logloss', balanced_logloss(pred,y_true), False

def original_binary_logloss_objective(y_pred, trn_data):
    y_true = trn_data.get_label()
    pred = 1/(1+np.exp(-y_pred))
    grad = pred - y_true
    hess = pred * (1-pred)
    return grad, hess

def evaluate(features,X_df,y_df):
    X_df_selected = X_df[features].copy()
    scores= []
    skf = StratifiedKFold(n_splits = 4, shuffle = True, random_state = 42)

    for tr_idx, va_idx in skf.split(X_df_selected, y_df):
        X_train, X_valid = X_df_selected.iloc[tr_idx], X_df_selected.iloc[va_idx]
        y_train, y_valid = y_df.iloc[tr_idx], y_df.iloc[va_idx]
        
        dtrain = lgb.Dataset(X_train.values, label = y_train.values)
        dvalid = lgb.Dataset(X_valid.values, label = y_valid.values)

        params = {'metric':'custom',
                'objective':'custom',
                'verbosity':-1,
                'random_state' : 42, 
                'learning_rate': 0.1,
                'early_stopping_round':20
                }

        callbacks = [lgb.early_stopping(20, verbose=0)]
        model = lgb.train(params,
                        dtrain, 
                        num_boost_round = 500,
                        valid_sets = [dtrain,dvalid],
                        valid_names = ['train','valid'],
                        feval = original_balanced_logloss_metric,
                        fobj = original_binary_logloss_objective,
                        callbacks = callbacks
                        )
        score = model.best_score['valid']['balanced_logloss']
        scores.append(score)
    score_cv = np.mean(scores)
    return score_cv