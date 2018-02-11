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
	arg('-dataset', '--dataset', type=str, help='which dataset')
	arg( '-nc','--nc', type=str, help="number of class\n")
	return vars(parser.parse_args())

par = read_params(sys.argv)

save_path = '../../DataSet/' + par['dataset'] + '/Data/' + par['dataset'] + '/Results/' + par['dataset'] + '_LR2'+ '.npy'
metrics = np.load(save_path).item() 
# ACC
measure = 'F1 macro'


baselines = list()
baselines_sd = list()
classifier = ['SVM', 'GB', 'RF', 'MB', 'LR', 'LR2']
for cf in classifier:
	key = cf + ': ' + measure
	baselines.append(round(np.average(metrics[key]),2))
	baselines_sd.append(round(np.std(metrics[key]),2))
print(baselines)
print(baselines_sd)

nb_size = [0, 50, 75, 100, 150, 200, 500, 1000, 2000, 3000]
nb_size = [0, 100]

MLP = dict()
MLP_sd = dict()
AD = ['A_ND', 'A_D']
iteration = [200]
for i in nb_size:
	for j in AD:
		for k in iteration:
			key1 = 'MLP' + ': ' + measure + '_' + str(j) + str(k)
			key = 'MLP' + ': ' + measure + '_' + str(j) + '_' + str(k) + '_' + str(i)
			if key1 not in MLP:
				MLP[key1] = list()
				MLP_sd[key1] = list()
				temp = metrics[key]
				temp1 = list()
				for val in temp:
					if val > 0.5:
						temp1.append(val)
				MLP[key1].append(round(np.average(temp1),2))
				MLP_sd[key1].append(round(np.std(temp1),2))
			else:
				#if i == nb_size[1]: 
					#print(metrics[key])
				temp = metrics[key]
				temp1 = list()
				for val in temp:
					if val > 0.5:
						temp1.append(val)
				MLP[key1].append(round(np.average(temp1),2))
				MLP_sd[key1].append(round(np.std(temp1),2))
				#MLP[key1].append(round(np.average(metrics[key]),3))

for j in AD:
	for k in iteration:
		key1 = 'MLP' + ': ' + measure + '_' + str(j) + str(k)
		print(key1)
		print(MLP[key1])
		print(MLP_sd[key1])
			#print (round(np.average(metrics[key]),3)) 


CNN = dict()
CNN_sd = dict()
AD = ['A_ND', 'A_D']
iteration = [200]
for i in nb_size:
	for j in AD:
		for k in iteration:
			key1 = 'CNN' + ': ' + measure + '_' + str(j) + str(k)
			key = 'CNN' + ': ' + measure + '_' + str(j) + '_' + str(k) + '_' + str(i)
			if key1 not in CNN:
				CNN[key1] = list()
				CNN_sd[key1] = list()
				temp = metrics[key]
				temp1 = list()
				for val in temp:
					if val > 0.5:
						temp1.append(val)
				CNN[key1].append(round(np.average(temp1),2))
				CNN_sd[key1].append(round(np.std(temp1),2))
			else:
				#if i == nb_size[1]: 
					#print(metrics[key])
				temp = metrics[key]
				temp1 = list()
				for val in temp:
					if val > 0.5:
						temp1.append(val)
				CNN[key1].append(round(np.average(temp1),2))
				CNN_sd[key1].append(round(np.std(temp1),2))
				#CNN[key1].append(round(np.average(metrics[key]),3))

for j in AD:
	for k in iteration:
		key1 = 'CNN' + ': ' + measure + '_' + str(j) + str(k)
		print(key1)
		print(CNN[key1])
		print(CNN_sd[key1])
			#print (round(np.average(metrics[key]),3)) 


