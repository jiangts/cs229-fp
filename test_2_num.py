import scipy.io as sio
import numpy as np
from sklearn.naive_bayes import *
from string import lowercase
import random

# created with the following octave/matlab command
# save -6 alpha_feature.mat feature_points_alpha    % MATLAB 6 compatible
a = sio.loadmat('num_feature.mat')
b = sio.loadmat('num_label.mat')

matrix = a['feature_points_num']
label = b['labels_num']

letters = list(lowercase)

num_samples, num_features =  matrix.shape

# shuffle the samples
temp_r = np.random.permutation(matrix.shape[0]);
matrix = matrix[temp_r,:]
label = label[:,temp_r]

matrix = list(matrix)
label = label.tolist()[0]
label = [chr(int(x)+47) for x in label]

gnb = GaussianNB()

print("Training = Testing Samples")
train_set = matrix
labels = label

y_pred = gnb.fit(train_set, labels).predict(train_set)

print("Number of mislabeled points out of a total %d points : %d"  % (len(train_set) ,(labels != y_pred).sum()))

labels_fail = [0] * 10
labels_appear = [0] * 10
for iteration in range(250):
	temp_r = np.random.permutation(num_samples);
	
	combined = zip(matrix, label)
	random.shuffle(combined)
	matrix[:], label[:] = zip(*combined)

	print("\nTraining != Testing Samples")
	num_training = int(round(0.9*num_samples))
	num_testing =  num_samples - num_training
	train_set = matrix[1:num_training]
	train_labels = labels[1:num_training]

	test_set = matrix[num_training:num_samples]
	test_labels = labels[num_training:num_samples]


	y_pred = gnb.fit(train_set, train_labels).predict(test_set)

	print("Number of mislabeled points out of a total %d points : %d"  % (len(test_set) ,(test_labels != y_pred).sum()))

	for (pred,act) in zip(y_pred,test_labels):
		if act!=pred:
			print act,pred

	denom = np.array([0]*10, dtype=np.float)
	num = np.array([0]*10, dtype=np.float)
	for i in range(len(test_labels)):
		letter_index = ord(test_labels[i])-ord('0')
		denom[letter_index] += 1
		if test_labels[i]!=y_pred[i]:
			labels_fail[letter_index] += 1	
		labels_appear[letter_index] += 1

labels_appear = [1 if x==0 else x for x in labels_appear]

print [float(labels_fail[x])/labels_appear[x] for x in range(10)]
