import pandas as pd
import os

def load_data():
     
    working_dir = '/Users/smokey/COGS118A/EEG/' 
    train_path = os.path.join(working_dir, 'new.csv')
    
    data = pd.read_csv(train_path, header=None)
    r, c = data.shape

    y = data[data.columns[c-1]]
    print(y.shape)
    print(y.head())
    print(y.dtypes)
    X = data[data.columns[0:c-1]]

    print(X.shape)
    print(X.head())
    print(X.dtypes)
    return X, y

