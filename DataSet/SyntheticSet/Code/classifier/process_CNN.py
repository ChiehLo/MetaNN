from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import argparse as ap
import sys

def read_params(args):
	parser = ap.ArgumentParser(description='Specify the probability')
	arg = parser.add_argument
	arg( '-p1','--p1', type=str, help="p1\n")
	arg( '-p2','--p2', type=str, help="p2\n")
	arg( '-p3','--p3', type=str, help="p3\n")
	arg( '-nc','--nc', type=str, help="number of class\n")
	arg( '-configure','--configure', type=str, help="configuration of microbial compositions\n")
	return vars(parser.parse_args())


def read_files(IT, p1, p2, p3, n_class, configure):
	postfix = '_' + str(p1) + '_' + str(p2) + '_' + str(p3) + '_' + str(n_class) + '_' + str(configure)
	prefix = '/home/chiehlo/Desktop/deep/Synthetic/RealData/IT' + str(IT) 
	train_sample_prefix = prefix + '/TrainSampleSynthetic_' + str(IT) + postfix +'.txt'
	train_label_prefix = prefix + '/TrainLabelSynthetic_' + str(IT) + postfix + '.txt'
	train_sample = np.loadtxt(train_sample_prefix, delimiter='\t')
	train_label = np.loadtxt(train_label_prefix, delimiter='\t').astype(int)
	row_sums = train_sample.sum(axis=1)
	row_sums = row_sums[:, np.newaxis]
	row_sums[np.where(row_sums == 0.0)] = 1 
	train_sample = train_sample / row_sums
	#row_sums = train_sample.sum(axis=1)
	#train_sample = train_sample / row_sums[:, np.newaxis]
	print(train_sample.shape[1])

	test_sample_prefix = prefix + '/TestSampleSynthetic_' + str(IT) + postfix + '.txt'
	test_label_prefix = prefix + '/TestLabelSynthetic_' + str(IT) + postfix + '.txt'
	test_sample = np.loadtxt(test_sample_prefix, delimiter='\t')
	test_label = np.loadtxt(test_label_prefix, delimiter='\t')
	row_sums = test_sample.sum(axis=1)
	row_sums = row_sums[:, np.newaxis]
	row_sums[np.where(row_sums == 0.0)] = 1 
	test_sample = test_sample / row_sums
	#row_sums = test_sample.sum(axis=1)
	#test_sample = test_sample / row_sums[:, np.newaxis]


	train_label = train_label.ravel()
	train_sample, train_label = shuffle(train_sample, train_label, random_state=1)

	return train_sample, train_label, test_sample, test_label

def evaluate_metrics_CNN(test_sample, test_label, test_label_matrix, it, run_times, 
	AD, iteration, p1, p2, p3, n_class, configure, result, CNN_flag = True):
	if CNN_flag == True:
		prefix = '/home/chiehlo/Desktop/deep/Synthetic/Results_CNN/' + str(run_times) 
		cf = 'CNN'

	con_path = '_' + str(p1) + '_' + str(p2) + '_' + str(p3) + '_' + str(n_class) + '_' + str(configure)
	postfix = '/IT' + str(it) + '_A_' + AD + '_' + str(iteration) + '_10' + con_path  + '.txt'
	path = prefix + postfix
	ACC_result = np.loadtxt(path, delimiter='\t') 
	#print(path)
	# ACC
	for i in xrange(1):
		#print(i)
		key = cf + ': ' + 'ACC_' + str(AD) + '_' + str(iteration) + con_path
		if key not in result:
			result[key] = list()
		result[key].append(round(ACC_result,3))
	# F1
	for i in xrange(1):
		postfix = '/prob' + '/IT' + str(it) + '_A_' + str(AD) + '_Prob_' + str(iteration) + '_10' + con_path + '.txt'  
		prob_read = np.loadtxt(prefix + postfix, delimiter='\t')
		y_pred =  np.argmax(prob_read, axis=1)
		# F1 macro
		key = cf + ': ' + 'F1 macro_' + str(AD) + '_' + str(iteration) + con_path
		if key not in result:
			result[key] = list()
		result[key].append(round(f1_score(test_label, y_pred, average='macro'), 3))
		# F1 micro
		key = cf + ': ' + 'F1 micro_' + str(AD) + '_' + str(iteration) + con_path
		if key not in result:
			result[key] = list()
		result[key].append(round(f1_score(test_label, y_pred, average='micro'),3))
		### roc_auc
		roc_auc = np.zeros([int(n_class), 1])
		fpr = dict()
		tpr = dict()
		key = cf + ': ' + 'AUC macro_' + str(AD) + '_' + str(iteration) + con_path
		if key not in result:
			result[key] = list()
		for j in range(int(n_class)):
			fpr[j], tpr[j], _ = roc_curve(test_label_matrix[:, j], prob_read[:, j])
			roc_auc[j] = auc(fpr[j], tpr[j])
		#print(np.average(roc_auc))
		result[key].append(round(np.average(roc_auc),3)) 
		# ROC micro
		key = cf + ': ' + 'AUC micro_' + str(AD) + '_' + str(iteration) + con_path
		if key not in result:
			result[key] = list()
		fpr["micro"], tpr["micro"], _ = roc_curve(test_label_matrix.ravel(), prob_read[:, 0:(int(n_class))].ravel())
		result[key].append(round(auc(fpr["micro"], tpr["micro"]),3))
	
	return result

par = read_params(sys.argv)

IT = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
run_times = 5
ACC = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
n_classes = int(par['nc'])
fpr = dict()
tpr = dict()
roc_auc = dict()
roc_auc["micro"] = 0
metrics = dict()
prob_classifier = ['CNN']
for it in xrange(len(IT)):
	train_sample, train_label, test_sample, test_label = read_files(IT[it], p1 = par['p1'], p2 = par['p2'], p3 = par['p3'], n_class = par['nc'], configure = par['configure'])
	temp = np.zeros((len(test_label),n_classes))
	for i in range(len(test_label)):
		temp[i,int(test_label[i])] = 1
	AD = 'ND'
	for i in xrange(run_times):
		metrics = evaluate_metrics_CNN(test_sample = test_sample, test_label = test_label, test_label_matrix = temp, 
			it = (it+1), run_times = (i+1), AD = AD, iteration = 200, p1 = par['p1'], p2 = par['p2'], p3 = par['p3'], n_class = par['nc'], configure = par['configure'], result = metrics)
		


print(metrics)
#postfix = par['p1'] + '_' + par['p2'] + '_' + par['p3'] + '_' + par['nc'] + '_' + par['configure']
#save_path = '/Users/chiehlo/Desktop/HMP_project/deep/datasets/SyntheticSet/Data/Results/' + postfix + '.npy'
#np.save(save_path, metrics) 




