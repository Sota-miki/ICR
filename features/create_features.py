import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import itertools
from sklearn.decomposition import PCA

dataset_df = pd.read_csv(r"C:\Users\ghibl\ICR\data\input\processed_train.csv")
dataset_df.drop("Id",axis = 1, inplace = True)

X_df = dataset_df.drop("Class",axis =1)
features_df = X_df.copy()

#Standardize
num_cols = list(features_df.columns)
num_cols = [i for i in num_cols if not i =="EJ"]
sc = StandardScaler()
features_df[num_cols] = sc.fit_transform(features_df[num_cols])

#PCA
pca = PCA(n_components = 5)
pca_df = pd.DataFrame(pca.fit_transform(features_df),
                      columns = ["Component1","Component2","Component3","Component4","Component5"])
features_df = pd.concat([features_df,pca_df],axis = 1)



#create new features saving added and multiplied values for each feature
columns = list(features_df.columns)
com_columns = list(itertools.combinations(columns, 2))
for com_col in com_columns:
    features_df[f"{com_col[0]}_{com_col[1]}_mul"] = X_df[f"{com_col[0]}"] *  X_df[f"{com_col[1]}"]
    features_df[f"{com_col[0]}_{com_col[1]}_sum"] = X_df[f"{com_col[0]}"] +  X_df[f"{com_col[1]}"]

