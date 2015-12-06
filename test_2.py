import scipy.io as sio
import numpy as np
from sklearn.naive_bayes import *
from sklearn import svm
from string import lowercase
import random

# created with the following octave/matlab command
# save -6 alpha_feature.mat feature_points_alpha    % MATLAB 6 compatible
a = sio.loadmat('alpha_feature.mat')
b = sio.loadmat('alpha_label.mat')

matrix = a['feature_points_alpha']
label = b['labels_alpha']

#for i in range(len(label[0])):
#	x = chr(label[0][i]+96)
#	if x=='b':
#		let = 'e'
#	elif x=='c':
#		let = 'e'
#	elif x=='d':
#		let = 'e'
#	elif x=='g':
#		let = 'e'
#	elif x=='p':
#		let = 'e'
#	elif x=='t':
#		let = 'e'
#	elif x=='v':
#		let = 'e'
#	elif x=='z':
#		let = 'e'
#
#	elif x=='n':
#		let = 'm'
#		
#	elif x=='s':
#		let = 'f'
#	elif x=='x':
#		let = 'f'
#
#
#	elif x=='j':
#		let = 'a'
#	elif x=='k':
#		let = 'a'
#
#	else:
#		let = x
#
#	label[0][i] = ord(let)-ord('a')+1

print label
letters = list(lowercase)

num_samples, num_features =  matrix.shape

# shuffle the samples
temp_r = np.random.permutation(matrix.shape[0]);
matrix = matrix[temp_r,:]
label = label[:,temp_r]

matrix = list(matrix)
label = label.tolist()[0]
label = [chr(int(x)+96) for x in label]

gnb = GaussianNB()
clf = svm.SVC(kernel='linear')

print("Training = Testing Samples")
train_set = matrix
labels = label

y_pred = gnb.fit(train_set, labels).predict(train_set)

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

print letters

labels_fail = [0] * labels_num
labels_appear = [0] * labels_num
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
		if act!=pred and act=='z':
			print act,pred

	denom = np.array([0]*labels_num, dtype=np.float)
	num = np.array([0]*labels_num, dtype=np.float)
	for i in range(len(test_labels)):
		letter_index = ord(letters[ord(test_labels[i])-ord('a')])-ord('a')
		denom[letter_index] += 1
		if letters[ord(test_labels[i])-ord('a')]!=letters[ord(y_pred[i])-ord('a')]:
			labels_fail[letter_index] += 1	
		labels_appear[letter_index] += 1

labels_appear = [1 if x==0 else x for x in labels_appear]

print [float(labels_fail[x])/labels_appear[x] for x in range(labels_num)]
