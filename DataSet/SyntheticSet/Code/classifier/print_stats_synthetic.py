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
from os import listdir
from os.path import isfile, join

def read_params(args):
	parser = ap.ArgumentParser(description='Specify the probability')
	arg = parser.add_argument
	arg('-dataset', '--dataset', type=str, help='which dataset')
	arg( '-nc','--nc', type=str, help="number of class\n")
	arg('-configure', '--configure', type = str, help="configurations")
	arg('-type', '--type', type = str, help="types")
	return vars(parser.parse_args())

par = read_params(sys.argv)

mapping = ["0p0", "0p1", "0p2", "0p3", "0p4", "0p5", "0p6", "0p7", "0p8", "0p9", "1p0"]
types = "type" + par['type']
if types == "type1":
	P1 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	P2 = [1.0]
	P3 = [1.0]

if types == "type2":
	P1 = [1.0]
	P2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	P3 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

if types == "type3":
	P1 = [0.5, 0.7, 0.9]
	P2 = [0.5, 0.7, 0.9]
	P3 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

if types == "type4":
	P1 = [0.5]
	P2 = [0.5]
	P3 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

if types == "type5":
	P1 = [0.7]
	P2 = [0.5]
	P3 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

if types == "type6":
	P1 = [0.5]
	P2 = [0.7]
	P3 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

if types == "type7":
	P1 = [1.0]
	P2 = [0.1]
	P3 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

if types == "type8":
	P1 = [0.1]
	P2 = [1.0]
	P3 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

if types == "type8":
	P1 = [1]
	P2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	P3 = [0.5]

## heatmap
if types == "type10":
	P1 = [0.5, 0.7, 0.9]
	P2 = [0.5, 0.7, 0.9]
	P3 = [0.5]

ACC = dict()


heatmap = np.zeros([len(P2), len(P3)])
prefix = '/Users/chiehlo/Desktop/HMP_project/deep/datasets/SyntheticSet/Data/Results1/'
ACC = dict()
for p1 in P1:
	for p2 in P2:
		for p3 in P3:
			#print(p1)
			save_path = prefix + mapping[int(round(p1*10))] + '_' + mapping[int(round(p2*10))] + '_' + mapping[int(round(p3*10))] + '_' + par['nc'] + '_' + par['configure'] + '.npy'
			metrics = np.load(save_path).item() 
			# ACC
			measure = 'ACC'
			classifier = ['RF', 'GB', 'SVM', 'MB', 'MLP', 'LR', 'LR2']
			#classifier = ['RF', 'MB', 'MLP']
			#classifier = ['MLP']
			for cf in classifier:
				key = cf + ': ' + measure
				print (round(np.average(metrics[key]),3))
				if cf not in ACC:
					ACC[cf]  = list()
				ACC[cf].append(round(np.average(metrics[key]),3))
					 #heatmap[int(round(p2*10)-1), int(round(p3*10))] = round(np.average(metrics[key]),3)


prefix = '/Users/chiehlo/Desktop/HMP_project/deep/datasets/SyntheticSet/Data/CNN_process/'
for p1 in P1:
	for p2 in P2:
		for p3 in P3:
			save_path = prefix + mapping[int(round(p1*10))] + '_' + mapping[int(round(p2*10))] + '_' + mapping[int(round(p3*10))] + '_' + par['nc'] + '_' + par['configure'] + '.npy'
			metrics = np.load(save_path).item() 
			# ACC
			measure = 'ACC'
			classifier = ['CNN']
			#classifier = ['RF', 'MB', 'MLP']
			#classifier = ['MLP']
			for cf in classifier:
				key = cf + ': ' + measure + '_ND_200_' + mapping[int(round(p1*10))] + '_' + mapping[int(round(p2*10))] + '_' + mapping[int(round(p3*10))] + '_' + par['nc'] + '_' + par['configure']
				count = 0.0
				sumup = 0.0
				#print(metrics[key])
				for val in metrics[key]:
					if val > 0.5:
						count = count + 1
						sumup = sumup + val
				temp = sumup
				print (round(temp/count,3))
				if cf not in ACC:
					ACC[cf]  = list()
				ACC[cf].append(round(temp/count,3))
					 #heatmap[int(round(p2*10)-1), int(round(p3*10))] = round(np.average(metrics[key]),3)


print(ACC)

'''
P4 = [4, 2.3, 1.5, 1, 0.6, 0.4, 0.25]
PP = [4, 3, 2, 1, 0]
print(P4)
for cf in classifier:
	plt.plot(P4, ACC[cf], linewidth=3.0)
#plt.ylabel('some numbers')
plt.legend(classifier)
plt.ylim((0.7,1.0))
plt.xticks(PP, fontsize=20)
plt.yticks(fontsize=20)
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('legend.pdf')
plt.savefig(pp, format='pdf')
pp.close()
plt.show()

#print(heatmap)
#plt.imshow(heatmap, cmap='hot', interpolation='nearest')
#plt.show()

'''
