import scipy.io as sio
import numpy as np
from sklearn.naive_bayes import *
from sklearn import svm
from string import lowercase
import random

# Theano
# https://github.com/gwtaylor/theano-rnn/blob/master/basic_rnn_example.py

# PyBrain
# http://stackoverflow.com/questions/25967922/pybrain-time-series-prediction-using-lstm-recurrent-nets
# https://github.com/pybrain/pybrain/blob/master/examples/supervised/neuralnets%2Bsvm/example_rnn.py

# created with the following octave/matlab command
# save -6 alpha_feature.mat feature_points_alpha    % MATLAB 6 compatible
a = sio.loadmat('alpha_feature.mat')
b = sio.loadmat('alpha_label.mat')

matrix = a['feature_points_alpha']
label = b['labels_alpha']


print(label)
letters = list(lowercase)

num_samples, num_features = matrix.shape

# shuffle the samples
temp_r = np.random.permutation(matrix.shape[0]);
matrix = matrix[temp_r,:]
label = label[:,temp_r]

matrix = list(matrix)
label = label.tolist()[0]
label = [chr(int(x)+96) for x in label]


print("Training = Testing Samples")
train_set = matrix
labels = label

### from pybrain.datasets import SequentialDataSet
### from itertools import cycle
### 
### data = [1] * 3 + [2] * 3
### data *= 3
### print(data)
### 
### 
### ds = SequentialDataSet(1, 1)
### for sample, next_sample in zip(data, cycle(data[1:])):
###     ds.addSample(sample, next_sample)
### 
### from pybrain.tools.shortcuts import buildNetwork
### from pybrain.structure.modules import LSTMLayer
### 
### net = buildNetwork(1, 5, 1, 
###                    hiddenclass=LSTMLayer, outputbias=False, recurrent=True)
### 
### from pybrain.supervised import RPropMinusTrainer
### from sys import stdout
### 
### trainer = RPropMinusTrainer(net, dataset=ds)
### train_errors = [] # save errors for plotting later
### EPOCHS_PER_CYCLE = 5
### CYCLES = 100
### EPOCHS = EPOCHS_PER_CYCLE * CYCLES
### for i in xrange(CYCLES):
###     trainer.trainEpochs(EPOCHS_PER_CYCLE)
###     train_errors.append(trainer.testOnData())
###     epoch = (i+1) * EPOCHS_PER_CYCLE
###     print("\r epoch {}/{}".format(epoch, EPOCHS), end="")
###     stdout.flush()
### 
### print()
### print("final error =", train_errors[-1])

from sknn.mlp import Classifier, Layer

train_set = np.matrix(train_set)
labels = np.matrix(labels)

nn = Classifier(
    layers=[
        Layer("Maxout", units=100, pieces=2),
        Layer("Softmax")],
    learning_rate=0.001,
    n_iter=100)
nn.fit(train_set, np.transpose(labels))

y_pred = np.transpose(nn.predict(np.matrix(train_set)))

print("Number of mislabeled points out of a total %d points : %d"  % (len(train_set) ,(labels != y_pred).sum()))

labels_num = 26

letters = list(lowercase)
for i in range(26):
  x = letters[i]
  if x=='b':
    letters[i] = 'e'
  elif x=='c':
    letters[i] = 'e'
  elif x=='d':
    letters[i] = 'e'
  elif x=='g':
    letters[i] = 'e'
  elif x=='p':
    letters[i] = 'e'
  elif x=='t':
    letters[i] = 'e'
  elif x=='v':
    letters[i] = 'e'
  elif x=='z':
    letters[i] = 'e'

  elif x=='j':
    letters[i] = 'a'
  elif x=='k':
    letters[i] = 'a'

  elif x=='r':
    letters[i] = 'i'

  elif x=='n':
    letters[i] = 'm'

  elif x=='o':
    letters[i] = 'l'

  elif x=='h':
    letters[i] = 'f'

  elif x=='x':
    letters[i] = 's'

  elif x=='q':
    letters[i] = 'u'

print(letters)

labels_fail = [0] * labels_num
labels_appear = [0] * labels_num


# for iteration in range(250):
#   temp_r = np.random.permutation(num_samples);
# 
#   combined = zip(matrix, label)
#   random.shuffle(combined)
#   matrix[:], label[:] = zip(*combined)
# 
#   print("\nTraining != Testing Samples")
#   num_training = int(round(0.9*num_samples))
#   num_testing =  num_samples - num_training
#   train_set = matrix[1:num_training]
#   train_labels = labels[1:num_training]
# 
#   test_set = matrix[num_training:num_samples]
#   test_labels = labels[num_training:num_samples]
# 
#   y_pred = algo.fit(train_set, train_labels).predict(test_set)
# 
#   print("Number of mislabeled points out of a total %d points : %d",  % (len(test_set) ,(test_labels != y_pred).sum()))
# 
#   for (pred,act) in zip(y_pred,test_labels):
#     if act!=pred and act=='z':
#       print(act,pred)
# 
#   denom = np.array([0]*labels_num, dtype=np.float)
#   num = np.array([0]*labels_num, dtype=np.float)
#   for i in range(len(test_labels)):
#     letter_index = ord(letters[ord(test_labels[i])-ord('a')])-ord('a')
#     denom[letter_index] += 1
#     if letters[ord(test_labels[i])-ord('a')]!=letters[ord(y_pred[i])-ord('a')]:
#       labels_fail[letter_index] += 1	
#     labels_appear[letter_index] += 1

labels_appear = [1 if x==0 else x for x in labels_appear]

print([float(labels_fail[x])/labels_appear[x] for x in range(labels_num)])

