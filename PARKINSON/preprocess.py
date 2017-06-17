import pandas as pd
import os
import numpy as np


working_dir = '/Users/smokey/COGS118A/PARKINSON/'
input_file = 'raw_data/train_data.csv'
input_file2 = 'raw_data/test_data.csv'

output_file = 'raw_data/data.csv'
#train_file = 'train.csv'
#train_labels_file = 'train_labels.csv'
#test_file = 'test.csv'
#test_labels_file = 'test_labels.csv'

input_path = os.path.join(working_dir, input_file)
input_path2 = os.path.join(working_dir, input_file2)

output_path = os.path.join(working_dir, output_file)

#train_path = os.path.join(working_dir, train_file)
#train_labels_path = os.path.join(working_dir, train_labels_file)
#test_path = os.path.join(working_dir, test_file)
#test_labels_path = os.path.join(working_dir, test_labels_file)

data = pd.read_csv(input_path, header=None)
data2 = pd.read_csv(input_path2, header=None)

data.drop(data.columns[27], axis=1, inplace=True)
data.rename(columns={28:27}, inplace=True)


data[data.columns[21]] = data[data.columns[21]].astype('float64')
data[data.columns[25]] = data[data.columns[25]].astype('float64')
data[data.columns[20]] = data[data.columns[20]].astype('float64')

if ((data.dtypes == data2.dtypes).all()):
    new = pd.concat([data, data2])
    new.to_csv(output_path, header=False, index=False)
