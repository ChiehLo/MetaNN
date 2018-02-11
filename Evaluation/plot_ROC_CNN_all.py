## python2 ROC_CBH.py 
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

### MBSet 5 runs and 10 splits
RSEED = np.array([1,2,3,4,5])
IT = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
PP = np.array([0])
nInt = len(IT)
nSeed = len(RSEED)
nPP = len(PP)

EPOCH = 200
BATCH_SIZE = 32
n_classes = [6, 12]
datasets = ['CBH', 'CSS']
count1 = 0

for dataset in datasets:
	prefix =  '/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/' + dataset + '/Splits/IT' 
	prefix_save = '/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/' + dataset + '/Results_CNN/' 
	print(dataset)
	print(n_classes[count1])
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	tprMicro = []
	fprMicro = []
	aucMicro = []
	count = 0
	for seed in range(nSeed):	
		for it in range(len(IT)):
			test_path = prefix + str(IT[it]) +  '/TestLabel_' + str(IT[it])  + '.txt'
			test_label = np.loadtxt(test_path, delimiter='\t')
			temp = np.zeros((len(test_label),n_classes[count1]))
			for i in range(len(test_label)):
				temp[i,int(test_label[i])] = 1

			for pp in range(len(PP)):
				save_path = prefix_save + str(RSEED[seed])  + '/prob/IT' + str(IT[it]) + '_A_D_Prob_' + str(EPOCH) + '_' + str(BATCH_SIZE) + '_' + str(PP[pp]) + '.txt'
				test_prob = np.loadtxt(save_path, delimiter='\t')
				test_prob = test_prob[:,0:(n_classes[count1])]
				# compute micro ROC
				fpr_tmp, tpr_tmp, _ = roc_curve(temp.ravel(), test_prob.ravel())
				tprMicro.append(interp(mean_fpr, fpr_tmp, tpr_tmp))
				fprMicro.append(fpr_tmp)
				tprMicro[-1][0] = 0.0
				roc_auc = auc(fpr_tmp, tpr_tmp)
				aucMicro.append(roc_auc)
				count = count + 1

		mean_tpr = np.mean(tprMicro, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucMicro)
		print(mean_auc)

	tprPP = []
	for i in xrange(len(PP)):
		for seed in xrange(nSeed):
			for j in xrange(len(IT)):
				tprPP.append(tprMicro[j*len(PP)+ seed*len(PP)*len(IT) + i][:])
				#plt.plot(mean_fpr, tprMicro[j*len(PP)+ seed*len(PP)*len(IT) + i][:], lw=1, alpha=0.3)
		mean_tpr = np.mean(tprPP, axis=0)
		std_tpr = np.std(tprPP, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1)	
		plt.plot(mean_fpr, mean_tpr, lw=4, alpha=1, label='%s (AUC = %0.2f)' % (dataset, roc_auc))
		tprPP = []
	count1 = count1 + 1


## HMP 10 splits 5 runs
RSEED = np.array([1,2,3,4,5])
IT = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
PP = np.array([0])
nInt = len(IT)
nSeed = len(RSEED)
nPP = len(PP)

EPOCH = 100
BATCH_SIZE = 32
n_classes = [5]
datasets = ['HMP']
names = ['HMP']
count1 = 0

for dataset in datasets:
	prefix =  '/Users/chiehlo/Desktop/HMP_project/deep/datasets/' + names[count1] + '/Data' + '/Splits/IT' 
	prefix_save = '/Users/chiehlo/Desktop/HMP_project/deep/datasets/' + names[count1] + '/Data'  + '/Results/' 
	print(dataset)
	print(n_classes[count1])
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	tprMicro = []
	fprMicro = []
	aucMicro = []
	count = 0
	for seed in range(nSeed):	
		for it in range(len(IT)):
			test_path = prefix + str(IT[it]) +  '/TestLabelHMP_' + str(IT[it])  + '.txt'
			test_label = np.loadtxt(test_path, delimiter='\t')
			temp = np.zeros((len(test_label),n_classes[count1]))
			for i in range(len(test_label)):
				temp[i,int(test_label[i])] = 1

			for pp in range(len(PP)):
				save_path = prefix_save + str(RSEED[seed])  + '/prob/IT' + str(IT[it]) + '_A_D_Prob_' + str(EPOCH) + '_' + str(BATCH_SIZE) + '_' + str(PP[pp]) + '.txt'
				test_prob = np.loadtxt(save_path, delimiter='\t')
				test_prob = test_prob[:,0:(n_classes[count1])]
				# compute micro ROC
				fpr_tmp, tpr_tmp, _ = roc_curve(temp.ravel(), test_prob.ravel())
				tprMicro.append(interp(mean_fpr, fpr_tmp, tpr_tmp))
				fprMicro.append(fpr_tmp)
				tprMicro[-1][0] = 0.0
				roc_auc = auc(fpr_tmp, tpr_tmp)
				aucMicro.append(roc_auc)
				count = count + 1

		mean_tpr = np.mean(tprMicro, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucMicro)
		print(mean_auc)

	tprPP = []
	for i in xrange(len(PP)):
		for seed in xrange(nSeed):
			for j in xrange(len(IT)):
				tprPP.append(tprMicro[j*len(PP)+ seed*len(PP)*len(IT) + i][:])
				#plt.plot(mean_fpr, tprMicro[j*len(PP)+ seed*len(PP)*len(IT) + i][:], lw=1, alpha=0.3)
		mean_tpr = np.mean(tprPP, axis=0)
		std_tpr = np.std(tprPP, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1)	
		plt.plot(mean_fpr, mean_tpr, lw=4, alpha=1, label='%s (AUC = %0.2f)' % (dataset, roc_auc))
		tprPP = []
	count1 = count1 + 1

### MBSet 5 runs and 10 splits
RSEED = np.array([1,2,3,4,5])
IT = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
PP = np.array([0])
nInt = len(IT)
nSeed = len(RSEED)
nPP = len(PP)

EPOCH = 200
BATCH_SIZE = 32
n_classes = [7, 3, 6]
datasets = ['CS', 'FS', 'FSH']
count1 = 0

for dataset in datasets:
	prefix =  '/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/' + dataset + '/Splits/IT' 
	prefix_save = '/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/' + dataset + '/Results_CNN/' 
	print(dataset)
	print(n_classes[count1])
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	tprMicro = []
	fprMicro = []
	aucMicro = []
	count = 0
	for seed in range(nSeed):	
		for it in range(len(IT)):
			test_path = prefix + str(IT[it]) +  '/TestLabel_' + str(IT[it])  + '.txt'
			test_label = np.loadtxt(test_path, delimiter='\t')
			temp = np.zeros((len(test_label),n_classes[count1]))
			for i in range(len(test_label)):
				temp[i,int(test_label[i])] = 1

			for pp in range(len(PP)):
				save_path = prefix_save + str(RSEED[seed])  + '/prob/IT' + str(IT[it]) + '_A_D_Prob_' + str(EPOCH) + '_' + str(BATCH_SIZE) + '_' + str(PP[pp]) + '.txt'
				test_prob = np.loadtxt(save_path, delimiter='\t')
				test_prob = test_prob[:,0:(n_classes[count1])]
				# compute micro ROC
				fpr_tmp, tpr_tmp, _ = roc_curve(temp.ravel(), test_prob.ravel())
				tprMicro.append(interp(mean_fpr, fpr_tmp, tpr_tmp))
				fprMicro.append(fpr_tmp)
				tprMicro[-1][0] = 0.0
				roc_auc = auc(fpr_tmp, tpr_tmp)
				aucMicro.append(roc_auc)
				count = count + 1

		mean_tpr = np.mean(tprMicro, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucMicro)
		print(mean_auc)

	tprPP = []
	for i in xrange(len(PP)):
		for seed in xrange(nSeed):
			for j in xrange(len(IT)):
				tprPP.append(tprMicro[j*len(PP)+ seed*len(PP)*len(IT) + i][:])
				#plt.plot(mean_fpr, tprMicro[j*len(PP)+ seed*len(PP)*len(IT) + i][:], lw=1, alpha=0.3)
		mean_tpr = np.mean(tprPP, axis=0)
		std_tpr = np.std(tprPP, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1)	
		plt.plot(mean_fpr, mean_tpr, lw=4, alpha=1, label='%s (AUC = %0.2f)' % (dataset, roc_auc))
		tprPP = []
	count1 = count1 + 1

## IBD and PDX 5 splits 5 runs
RSEED = np.array([1,2,3,4,5])
IT = np.array([1, 2, 3, 4, 5])
PP = np.array([0])
nInt = len(IT)
nSeed = len(RSEED)
nPP = len(PP)

EPOCH = 200
BATCH_SIZE = 32
n_classes = [2, 4]
datasets = ['IBD', 'PDX']
names = ['IBDSet', 'PDX']
count1 = 0

for dataset in datasets:
	prefix =  '/Users/chiehlo/Desktop/HMP_project/deep/datasets/' + names[count1] + '/Data/' + dataset + '/Splits/IT' 
	prefix_save = '/Users/chiehlo/Desktop/HMP_project/deep/datasets/' + names[count1] + '/Data/' + dataset + '/Results_CNN/' 
	print(dataset)
	print(n_classes[count1])
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	tprMicro = []
	fprMicro = []
	aucMicro = []
	count = 0
	for seed in range(nSeed):	
		for it in range(len(IT)):
			test_path = prefix + str(IT[it]) +  '/TestLabel_' + str(IT[it])  + '.txt'
			test_label = np.loadtxt(test_path, delimiter='\t')
			temp = np.zeros((len(test_label),n_classes[count1]))
			for i in range(len(test_label)):
				temp[i,int(test_label[i])] = 1

			for pp in range(len(PP)):
				save_path = prefix_save + str(RSEED[seed])  + '/prob/IT' + str(IT[it]) + '_A_D_Prob_' + str(EPOCH) + '_' + str(BATCH_SIZE) + '_' + str(PP[pp]) + '.txt'
				test_prob = np.loadtxt(save_path, delimiter='\t')
				test_prob = test_prob[:,0:(n_classes[count1])]
				# compute micro ROC
				fpr_tmp, tpr_tmp, _ = roc_curve(temp.ravel(), test_prob.ravel())
				tprMicro.append(interp(mean_fpr, fpr_tmp, tpr_tmp))
				fprMicro.append(fpr_tmp)
				tprMicro[-1][0] = 0.0
				roc_auc = auc(fpr_tmp, tpr_tmp)
				aucMicro.append(roc_auc)
				count = count + 1

		mean_tpr = np.mean(tprMicro, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucMicro)
		print(mean_auc)

	tprPP = []
	for i in xrange(len(PP)):
		for seed in xrange(nSeed):
			for j in xrange(len(IT)):
				tprPP.append(tprMicro[j*len(PP)+ seed*len(PP)*len(IT) + i][:])
				#plt.plot(mean_fpr, tprMicro[j*len(PP)+ seed*len(PP)*len(IT) + i][:], lw=1, alpha=0.3)
		mean_tpr = np.mean(tprPP, axis=0)
		std_tpr = np.std(tprPP, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1)	
		plt.plot(mean_fpr, mean_tpr, lw=4, alpha=1, label='%s (AUC = %0.2f)' % (dataset, roc_auc))
		tprPP = []
	count1 = count1 + 1





plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('./MLP_v1.pdf')
plt.savefig(pp, format='pdf')
pp.close()
plt.legend(loc="lower right")
plt.legend(frameon=False)
pp = PdfPages('./MLP_legend_v1.pdf')
plt.savefig(pp, format='pdf')
pp.close()
plt.show()



