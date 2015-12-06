import scipy.io as sio
import numpy as np
from sklearn.naive_bayes import *
from string import lowercase

# created with the following octave/matlab command
# save -6 alpha_feature.mat feature_points_alpha    % MATLAB 6 compatible
a = sio.loadmat('alpha_feature.mat')

matrix = a['feature_points_alpha']

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

	elif x=='h':
		letters[i] = 'a'
	elif x=='j':
		letters[i] = 'a'
	elif x=='k':
		letters[i] = 'a'

	elif x=='y':
		letters[i] = 'i'
	elif x=='o':
		letters[i] = 'i'
	elif x=='r':
		letters[i] = 'i'

	elif x=='f':
		letters[i] = 'm'
	elif x=='l':
		letters[i] = 'm'
	elif x=='n':
		letters[i] = 'm'
	elif x=='s':
		letters[i] = 'm'
	elif x=='x':
		letters[i] = 'm'

	elif x=='q':
		letters[i] = 'u'
	elif x=='w':
		letters[i] = 'u'

print letters

#access a sample (26 x 13) with matrix[:,:,1]
num_chars, num_samples, num_features =  matrix.shape

# shuffle the samples
for i in range(num_chars):
	matrix[i,:,:] = matrix[i,np.random.permutation(matrix.shape[1]),:]

gnb = GaussianNB()

print("Training = Testing Samples")
train_set = []
labels = []
for i in range(num_chars):
	for j in range(num_samples):
		next_sample = matrix[i,j,:]
		train_set.append(next_sample)
	labels += letters[i]*num_samples

print len(train_set[0])

y_pred = gnb.fit(train_set, labels).predict(train_set)

print("Number of mislabeled points out of a total %d points : %d"  % (len(train_set) ,(labels != y_pred).sum()))


print("\nTraining != Testing Samples")
num_training = 24
num_testing =  num_samples - num_training
train_set = []
train_labels = []
for i in range(num_chars):
	next_sample = list(matrix[i,0:num_training,:])
	train_set += next_sample
	train_labels += letters[i]*num_training

test_set = []
test_labels = []
for i in range(num_chars):
	next_sample = list(matrix[i,num_training:num_samples,:])
	test_set += next_sample
	test_labels += letters[i]*num_testing


y_pred = gnb.fit(train_set, train_labels).predict(test_set)

print("Number of mislabeled points out of a total %d points : %d"  % (len(test_set) ,(test_labels != y_pred).sum()))
print y_pred
print test_labels

denom = np.array([0]*26, dtype=np.float)
num = np.array([0]*26, dtype=np.float)
for i in range(len(test_labels)):
	letter_index = ord(test_labels[i])-ord('a')
	denom[letter_index] += 1
	if test_labels[i]!=y_pred[i]:
		num[letter_index] += 1

denom = [1 if x==0 else x for x in denom]
print num/denom
