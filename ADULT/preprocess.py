import pandas as pd
import os
import numpy as np


working_dir = '/Users/smokey/COGS118A/ADULT/'
input_file = 'raw_data/new_data.csv'
train_file = 'train.csv'
train_labels_file = 'train_labels.csv'
test_file = 'test.csv'
test_labels_file = 'test_labels.csv'

input_path = os.path.join(working_dir, input_file)
train_path = os.path.join(working_dir, train_file)
train_labels_path = os.path.join(working_dir, train_labels_file)
test_path = os.path.join(working_dir, test_file)
test_labels_path = os.path.join(working_dir, test_labels_file)

data = pd.read_csv(input_path, header=None)

data.rename(columns= {0:'age', 2:'fnlwgt', 4:'education-num', 10:'captial-gain'
                        ,11:'capital-loss',12:'hours-per-week',14:'labels'},inplace=True)

data.dropna(how='any', inplace=True)

for columns in data:
    if data[columns].dtype != np.dtype(object):
        data = data[data[columns].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
    
    elif data[columns].dtype == np.dtype(object):
        data = data[~data[columns].str.contains('\?')]
        data[columns] = data[columns].astype('category')
       
        if columns == 'labels':
            print(data[columns])
            data[columns] = data[columns].cat.codes
            print(data[columns])
        else:
          one_hot = pd.get_dummies(data[columns])
          data = data.drop(columns, axis=1)
          data = data.join(one_hot)

labels = data['labels'].astype('int64')

data.drop('labels', axis=1, inplace=True);

print(labels.dtypes)
#print(labels.head())
print(data.dtypes)
#print(data.head())
print(labels.shape)
print(data.shape)

length = data.shape[0] - 15000

test = data.iloc[length:, :]
print(test.shape)
labels_test = labels.iloc[length:]
print(labels_test.shape)

train = data.iloc[:30222,:]
print(train.shape)
labels_train = labels.iloc[:30222]
print(labels_train.shape)

train.to_csv(train_path, header=False, index=False)
labels_train.to_csv(train_labels_path, header=False, index=False)


test.to_csv(test_path, header=False, index=False)
labels_test.to_csv(test_labels_path, header=False, index=False)
