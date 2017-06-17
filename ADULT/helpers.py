import pandas as pd
import os

def load_data():
     
    working_dir = '/Users/smokey/COGS118A/ADULT/' 
    train_path = os.path.join(working_dir, 'new.csv')
    train_labels_path = os.path.join(working_dir, 'new_labels.csv') 
    
    X = pd.read_csv(train_path, header=None)
    y = pd.read_csv(train_labels_path, header=None)

    return X, y[0]   
