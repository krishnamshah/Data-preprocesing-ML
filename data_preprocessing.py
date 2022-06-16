import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

import datetime

#for cross-validation
from sklearn.model_selection import RepeatedStratifiedKFold

#seting seed and random_state for entire program
seed =45
random_state = 1234

#Preprocessing
#====================================================================
#importing the dataset






def preprocessing(data):
    # list of samples with no missing values
    null = (data.isnull().sum(axis=1) / len(data)).sort_values(ascending = False)
    list_samples = null[null<0.7].index.to_list() #0.7 is the percentage of missing values
    data = data.loc[list_sampples]
    
    
    #Separating target variable before preprocessing
    X = data.drop('target',axis=1)
    y = data['target']
    
    data = X.copy()
    #one-hot encoding for categorical data
    data = pd.get_dummies(data, columns = [])

    
    data_missing = (data.isnull().sum() / len(data)).sort_values(ascending = False)
    data_missing.head()
    
    # removing duplicate features
    data = data.T.drop_duplicates().T.copy()
    print("After deleting duplicate columns, data.shape: ", data.shape)

    #drop constant features
    data = drop_constant_feat(data)
    
    #data_c.shape columns
    out_col_list = outliers(data,0.25,0.75,1.5).index.to_list()
    data = data[data.columns.intersection(out_col_list)]
    print("data shape after removing outliers:",data.shape)

    #converting to float data type
    Xx = pd.DataFrame()
    for col in data.columns:
        try:
            Xx[col] = data[col].astype(float)
        except Exception:
            print(f"Failed to convert: {col}")
            
    #dropping null rows
    #data=data.dropna()
    #print(data.shape)
    # check if there missing data (this datasets do not show NAs
    # as we will see in the empty list output)
    
    data_X = Xx.copy()
    cor = data_X.corr()
    columns = np.full((cor.shape[0],), True, dtype=bool)
    for i in range(cor.shape[0]):
        for j in range(i+1, cor.shape[0]):
            if np.abs(cor.iloc[i,j]) >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = data_X.columns[columns]
    data_corr = data_X[selected_columns]
    print("Data shape after removing correlated features: ",data_corr.shape)
    
    X = data_corr.copy()

    print("Class Balance:\n",y.value_counts())


    return X,y

