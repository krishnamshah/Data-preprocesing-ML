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

#outliers
def outliers(df,q1,q2,thresh=1.5):
    print("Before shape: ", df.shape)
    Q1 = df.quantile(q1)
    Q3 = df.quantile(q2)
    IQR = Q3 - Q1
    out_r = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum(axis=0).sort_values(ascending = False)
    print(out_r)
    print("Free from outliers:",len(out_r[out_r<1]))
    return out_r[out_r<1]

#drops constant features	
def drop_constant_feat(df,thres = 0.10):
    print("Before: df shape:", df.shape)
    constant_filter = VarianceThreshold(threshold=0.10)
    constant_filter.fit_transform(df)
    constant_columns = [column for column in df.columns
                        if column not in df.columns[constant_filter.get_support()]]

    print("No. of constant columns:",len(constant_columns))
    constant_columns_to_remove = [i.strip() for i in constant_columns]
    df = df.drop(constant_columns_to_remove, axis=1)
    print("After: df shape:", df.shape)

    return df

#checks and remoevs 90% correlated features
def drop_corr_feat(data, thres = 0.9):
    """
    data : pandas dataframe
    """
    cor = data.corr()
    columns = np.full((cor.shape[0],), True, dtype=bool)
    for i in range(cor.shape[0]):
        for j in range(i+1, cor.shape[0]):
            if np.abs(cor.iloc[i,j]) >= thres:
                if columns[j]:
                    columns[j] = False
    selected_columns = data.columns[columns]
    data_corr = data[selected_columns]
    print("Data shape after removing correlated features: ",data_corr.shape)
    return data_corr

def preprocessing(data):
    """
    data = panda daataframe
    """
    # list of samples with no missing values
    null = (data.isnull().sum(axis=1) / len(data)).sort_values(ascending = False)
    list_samples = null[null<0.7].index.to_list() #0.7 is the percentage of missing values
    data = data.loc[list_sampples]
    
    
    #Separating target variable before preprocessing
    X = data.drop('target',axis=1)
    y = data['target']
    
    data = X.copy()
    #one-hot encoding for categorical data
    #data = pd.get_dummies(data, columns = [])

    #lists out columns with percentage...
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

    data = drop_corr_feat(Xx, 0.9)
    
    X = data_corr.copy()

    print("Class Balance:\n",y.value_counts())


    return X,y

X,y = preprocessing(df)

