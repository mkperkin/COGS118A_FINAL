from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf
from tensorflow.python.platform import gfile

import pandas as pd
import os
import numpy as np


def load_data():
  working_dir = '/Users/smokey/COGS118A/SENTIMENT/' 
  train_path = os.path.join(working_dir, 'train.txt')
  train_labels_path = os.path.join(working_dir, 'train_labels.txt') 

  
  bucket = [5, 10, 15, 20, 25]

  bucket1 = []
  labels1 = []

  bucket2 = []
  labels2 = []

  bucket3 = []
  labels3 = []
  
  bucket4 = []
  labels4 = []

  bucket5 = []
  labels5 = []

  with tf.gfile.GFile(train_path, mode="r") as source_file:
    with tf.gfile.GFile(train_labels_path, mode="r") as target_file:
      source = source_file.readline()
      target = target_file.readline()
      counter = 0
      
      while source:
        counter += 1
        if counter % 100 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        
        source_ids = [int(x) for x in source.split()]
        target = [int(n) for n in target.split()]
        
        length = len(source_ids)
        
        if (length <= bucket[0]):
            k = length
            while (k < bucket[0]):
                source_ids.append(0)
                k = k+1
            bucket1.append([source_ids])
            labels1.append(target)

        elif (len(source_ids) <= bucket[1]):
            k = length
            while (k < bucket[1]):
                source_ids.append(0)
                k = k+1
            bucket2.append([source_ids])
            labels2.append(target)

        elif (len(source_ids) <= bucket[2]):
            k = length
            while (k < bucket[2]):
                source_ids.append(0)
                k = k+1
            bucket3.append([source_ids])
            labels3.append(target)
        elif (len(source_ids) <= bucket[3]):            
            k = length
            while (k < bucket[3]):
                source_ids.append(0)
                k = k+1
            bucket4.append([source_ids])
            labels4.append(target)
        elif (len(source_ids) <= bucket[4]):
            k = length
            while (k < bucket[4]):
                source_ids.append(0)
                k = k+1
            bucket5.append([source_ids])
            labels5.append(target)

        source,target = source_file.readline(), target_file.readline()
  
  bucket1 = pd.DataFrame(np.array(bucket1).reshape(-1,bucket[0]))
  labels1 = np.array(labels1)
  labels1 = pd.DataFrame(labels1) 
  #labels1[labels1.columns[0]] = labels1[labels1.columns[0]].astype('int64')

  print(bucket1.shape)
  print(labels1.shape)
  bucket2 = pd.DataFrame(np.array(bucket2).reshape(-1,bucket[1]))
  labels2 = pd.DataFrame(labels2)
  print(bucket2.shape)
  print(labels2.shape)
  bucket3 = pd.DataFrame(np.array(bucket3).reshape(-1,bucket[2]))
  labels3 = pd.DataFrame(labels3)
  print(bucket3.shape)
  print(labels3.shape)
  bucket4 = pd.DataFrame(np.array(bucket4).reshape(-1,bucket[3]))
  labels4 = pd.DataFrame(labels4)
  print(bucket4.shape)
  print(labels4.shape)
  bucket5 = pd.DataFrame(np.array(bucket5).reshape(-1,bucket[4]))
  labels5 = pd.DataFrame(labels5)
  print(bucket5.shape)
  print(labels5.shape)
 
  r,c = bucket2.shape
  if(r % 10 != 0):
      offset = r % 10
      bucket2 = bucket2[:(r - offset)]
      labels2 = labels2[:(r-offset)]
      print(bucket2.shape)
      print(labels2.shape)
  return bucket2, labels2


load_data()

   # working_dir = '/Users/smokey/COGS118A/ADULT/' 
    #train_path = os.path.join(working_dir, 'new.csv')
    #train_labels_path = os.path.join(working_dir, 'new_labels.csv') 
    
    #X = pd.read_csv(train_path, header=None)
    #y = pd.read_csv(train_labels_path, header=None)

    #return X, y[0]   
