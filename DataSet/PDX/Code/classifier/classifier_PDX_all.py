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


def compute_feature_importance(el, feat, feat_sel, ltype):
	fi = feature_importance(feat, 0.0)
	if ltype == 'rf':
		t = el.feature_importances_
	elif (ltype == 'lasso') | (ltype == 'enet'):
		t = abs(el.coef_)/sum(abs(el.coef_))
	else:
		t = [1.0/len(feat_sel)]*len(feat_sel)
	ti = [feat.index(s) for s in feat_sel]
	fi.imp[ti] = t

	t = sorted(range(len(t)), key=lambda s: t[s], reverse=True)
	fi.feat_sel = [feat[ti[s]] for s in t if fi.imp[ti[s]] != 0]

	return fi


def fit_classifier(X, y, ctype, n_features, index):
	X_new = X[:,index[0:n_features]]
	if ctype == 'RF':
		clf = RandomForestClassifier(n_estimators=200).fit(X_new, y)
	elif ctype == 'SVM':
		hyper_parameters = [ {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']} ]
		clf = GridSearchCV(SVC(C=1, probability=True), hyper_parameters, cv=3, scoring='accuracy').fit(X_new, y)
	elif ctype == 'GB':
		clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1, max_depth=10, min_samples_split=5).fit(X_new, y)
	elif ctype == 'LR':
		clf = LogisticRegression(penalty='l1',random_state=0).fit(X_new, y)
	elif ctype == 'LR2':
		clf = LogisticRegression(penalty='l2',random_state=0).fit(X_new, y)
	elif ctype == 'MLP':
		clf = MLPClassifier(solver='adam', alpha=0.0001, max_iter=1000, learning_rate='adaptive', hidden_layer_sizes=(512,256,),  learning_rate_init=1e-3, activation='relu').fit(X_new, y)
	elif ctype == 'MB':
		clf = MultinomialNB(alpha=0.0001, fit_prior=True).fit(X_new, y)
	elif ctype == 'ENET':
		hyper_parameters = [np.logspace(-4, -0.5, 50), [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]]
		clf = ElasticNetCV(alphas=hyper_parameters[0], l1_ratio=hyper_parameters[1], cv=5).fit(X_new, y)
	return clf

def evaluate_metrics(clf, cf, test_sample, test_label, test_label_matrix, pred_label, n_class, result):
	# ACC
	key = cf + ': ' + 'ACC'
	if key not in result:
		result[key] = list()
	result[key].append(round(clf.score(test_sample,test_label),3))
	# F1 macro
	key = cf + ': ' + 'F1 macro'
	if key not in result:
		result[key] = list()
	result[key].append(round(f1_score(test_label, y_pred, average='macro'), 3))
	# F1 micro
	key = cf + ': ' + 'F1 micro'
	if key not in result:
		result[key] = list()
	result[key].append(round(f1_score(test_label, y_pred, average='micro'),3))


	###
	roc_auc = np.zeros([n_classes, 1])
	fpr = dict()
	tpr = dict()
	# ROC-AUC macro
	key = cf + ': ' + 'AUC macro'
	if key not in result:
		result[key] = list()
	test_prob = clf.predict_proba(test_sample)
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(test_label_matrix[:, i], test_prob[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	result[key].append(round(np.average(roc_auc),3)) 
	# ROC micro
	key = cf + ': ' + 'AUC micro'
	if key not in result:
		result[key] = list()
	fpr["micro"], tpr["micro"], _ = roc_curve(test_label_matrix.ravel(), test_prob.ravel())
	result[key].append(round(auc(fpr["micro"], tpr["micro"]),3))
	
	return result



def evaluate_metrics_MLP(test_sample, test_label, test_label_matrix, it, run_times, 
	AD, iteration, nb_size, dataset, n_classes, result, CNN_flag = False):
	if CNN_flag == True:
		prefix = '../../Data/' + dataset + '/Results_CNN/' + str(run_times) 
		cf = 'CNN'
	else:
		prefix = '../../Data/' + dataset + '/Results_MLP/' + str(run_times) 
		cf = 'MLP'
	postfix = '/IT' + str(it) + '_' + str(AD) + '_' + str(iteration) + '_32.txt'
	path = prefix + postfix
	ACC_result = np.loadtxt(path, delimiter='\t') 
	# ACC
	for i in xrange(len(ACC_result)):
		#print(i)
		key = cf + ': ' + 'ACC_' + str(AD) + '_' + str(iteration) + '_' + str(nb_size[i])
		if key not in result:
			result[key] = list()
		result[key].append(round(ACC_result[i],3))
	# F1
	for i in xrange(len(ACC_result)):
		postfix = '/prob' + '/IT' + str(it) + '_' + str(AD) + '_Prob_' + str(iteration) + '_32_' + str(nb_size[i]) + '.txt'  
		prob_read = np.loadtxt(prefix + postfix, delimiter='\t')
		y_pred =  np.argmax(prob_read, axis=1)
		# F1 macro
		key = cf + ': ' + 'F1 macro_' + str(AD) + '_' + str(iteration) + '_' + str(nb_size[i])
		if key not in result:
			result[key] = list()
		result[key].append(round(f1_score(test_label, y_pred, average='macro'), 3))
		# F1 micro
		key = cf + ': ' + 'F1 micro_' + str(AD) + '_' + str(iteration) + '_' + str(nb_size[i])
		if key not in result:
			result[key] = list()
		result[key].append(round(f1_score(test_label, y_pred, average='micro'),3))
		### roc_auc
		roc_auc = np.zeros([n_classes, 1])
		fpr = dict()
		tpr = dict()
		key = cf + ': ' + 'AUC macro_' + str(AD) + '_' + str(iteration) + '_' + str(nb_size[i])
		if key not in result:
			result[key] = list()
		for j in range(n_classes):
			fpr[j], tpr[j], _ = roc_curve(test_label_matrix[:, j], prob_read[:, j])
			roc_auc[j] = auc(fpr[j], tpr[j])
		#print(np.average(roc_auc))
		result[key].append(round(np.average(roc_auc),3)) 
		# ROC micro
		key = cf + ': ' + 'AUC micro_' + str(AD) + '_' + str(iteration) + '_' + str(nb_size[i])
		if key not in result:
			result[key] = list()
		fpr["micro"], tpr["micro"], _ = roc_curve(test_label_matrix.ravel(), prob_read[:, 0:(n_classes)].ravel())
		result[key].append(round(auc(fpr["micro"], tpr["micro"]),3))
	
	return result


def read_files(IT, dataset):
	prefix = '../../Data/' + dataset + '/Splits/IT' + str(IT) 
	train_sample_prefix = prefix + '/TrainSample_' + str(IT) + '.txt'
	train_label_prefix = prefix + '/TrainLabel_' + str(IT) + '.txt'
	train_sample = np.loadtxt(train_sample_prefix, delimiter='\t')
	train_label = np.loadtxt(train_label_prefix, delimiter='\t').astype(int)
	row_sums = train_sample.sum(axis=1)
	row_sums = row_sums[:, np.newaxis]
	row_sums[np.where(row_sums == 0.0)] = 1 
	train_sample = train_sample / row_sums


	test_sample_prefix = prefix + '/TestSample_' + str(IT) + '.txt'
	test_label_prefix = prefix + '/TestLabel_' + str(IT) + '.txt'
	test_sample = np.loadtxt(test_sample_prefix, delimiter='\t')
	test_label = np.loadtxt(test_label_prefix, delimiter='\t')
	row_sums = test_sample.sum(axis=1)
	row_sums = row_sums[:, np.newaxis]
	row_sums[np.where(row_sums == 0.0)] = 1 
	test_sample = test_sample / row_sums



	train_label = train_label.ravel()
	train_sample, train_label = shuffle(train_sample, train_label, random_state=1)

	return train_sample, train_label, test_sample, test_label

par = read_params(sys.argv)

IT = np.array([1, 2, 3, 4, 5])
run_times = 1
ACC = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
n_classes = int(par['nc'])
fpr = dict()
tpr = dict()
roc_auc = dict()
roc_auc["micro"] = 0
metrics = dict()
classifier = ['RF', 'SVM', 'GB', 'LR', 'MLP', 'MB']
prob_classifier = ['RF', 'GB']
nonProb_classifier = ['SVM', 'LR', 'MB', 'LR2']
for it in xrange(len(IT)):
	print(it)
	train_sample, train_label, test_sample, test_label = read_files(IT[it], dataset = par['dataset'])
	temp = np.zeros((len(test_label),n_classes))
	for i in range(len(test_label)):
		temp[i,int(test_label[i])] = 1
	run_times = 1
	for i in xrange(run_times):
		for cf in prob_classifier:
			clf = fit_classifier(X = train_sample, y = train_label, ctype = cf, n_features = train_sample.shape[1], index = np.arange(0,train_sample.shape[1]))
			y_pred = clf.predict(test_sample)
			y_pred = np.round(y_pred)
			
			metrics = evaluate_metrics(clf, cf, test_sample, test_label, temp, y_pred, n_classes, metrics)
	run_times = 1
	for i in xrange(run_times):
		AD = 'A_ND'
		nb_size = [0, 50, 75, 100, 150, 200, 400, 800, 1200]
		metrics = evaluate_metrics_MLP(test_sample = test_sample, test_label = test_label, test_label_matrix = temp, 
			it = (it+1), run_times = (i+1), AD = AD, iteration = 100, nb_size = nb_size, dataset = par['dataset'], n_classes = n_classes, result = metrics)
		metrics = evaluate_metrics_MLP(test_sample = test_sample, test_label = test_label, test_label_matrix = temp, 
			it = (it+1), run_times = (i+1), AD = AD, iteration = 200, nb_size = nb_size, dataset = par['dataset'], n_classes = n_classes, result = metrics)
		metrics = evaluate_metrics_MLP(test_sample = test_sample, test_label = test_label, test_label_matrix = temp, 
			it = (it+1), run_times = (i+1), AD = AD, iteration = 400, nb_size = nb_size, dataset = par['dataset'], n_classes = n_classes, result = metrics)
		metrics = evaluate_metrics_MLP(test_sample = test_sample, test_label = test_label, test_label_matrix = temp, 
			it = (it+1), run_times = (i+1), AD = AD, iteration = 200, nb_size = nb_size, dataset = par['dataset'], n_classes = n_classes, result = metrics, CNN_flag = True)
		metrics = evaluate_metrics_MLP(test_sample = test_sample, test_label = test_label, test_label_matrix = temp, 
			it = (it+1), run_times = (i+1), AD = AD, iteration = 400, nb_size = nb_size, dataset = par['dataset'], n_classes = n_classes, result = metrics, CNN_flag = True)
		AD = 'A_D'
		metrics = evaluate_metrics_MLP(test_sample = test_sample, test_label = test_label, test_label_matrix = temp, 
			it = (it+1), run_times = (i+1), AD = AD, iteration = 100, nb_size = nb_size, dataset = par['dataset'], n_classes = n_classes, result = metrics)
		metrics = evaluate_metrics_MLP(test_sample = test_sample, test_label = test_label, test_label_matrix = temp, 
			it = (it+1), run_times = (i+1), AD = AD, iteration = 200, nb_size = nb_size, dataset = par['dataset'], n_classes = n_classes, result = metrics)
		metrics = evaluate_metrics_MLP(test_sample = test_sample, test_label = test_label, test_label_matrix = temp, 
			it = (it+1), run_times = (i+1), AD = AD, iteration = 400, nb_size = nb_size, dataset = par['dataset'], n_classes = n_classes, result = metrics)
		metrics = evaluate_metrics_MLP(test_sample = test_sample, test_label = test_label, test_label_matrix = temp, 
			it = (it+1), run_times = (i+1), AD = AD, iteration = 200, nb_size = nb_size, dataset = par['dataset'], n_classes = n_classes, result = metrics, CNN_flag = True)
		metrics = evaluate_metrics_MLP(test_sample = test_sample, test_label = test_label, test_label_matrix = temp, 
			it = (it+1), run_times = (i+1), AD = AD, iteration = 400, nb_size = nb_size, dataset = par['dataset'], n_classes = n_classes, result = metrics, CNN_flag = True)
		
	
	for cf in nonProb_classifier:
		clf = fit_classifier(X = train_sample, y = train_label, ctype = cf, n_features = train_sample.shape[1], index = np.arange(0,train_sample.shape[1]))
		y_pred = clf.predict(test_sample)
		y_pred = np.round(y_pred)
			
		metrics = evaluate_metrics(clf, cf, test_sample, test_label, temp, y_pred, n_classes, metrics)
	


print(metrics)
save_path = '../../Data/' + par['dataset'] + '/Results/' + par['dataset'] + '.npy'
np.save(save_path, metrics) 



