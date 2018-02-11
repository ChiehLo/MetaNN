from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np
import pandas as pd
#import seaborn

# Variables to call
otuinfile = '../Data/PDX.csv'
# Column name from mapfile for disease
groups = ['Psoriasis Lesion', 'Psoriasis Normal', 'Control']

# Data read
#data = pd.read_csv(otuinfile,index_col=0,skiprows=1)
data = pd.read_csv(otuinfile,index_col=0)
#print(data)
hh = pd.read_csv(otuinfile, index_col=0, header=None,nrows=1).transpose()
#hh = hh.replace(to_replace=['Psoriasis Lesion', 'Psoriasis Normal', 'Control'], value=['disorder','disorder','no-disorder'])
hh = pd.DataFrame.as_matrix(hh).ravel()

encoder = LabelEncoder()
y = pd.Series(encoder.fit_transform(hh))


otutab_trans = data.transpose()
X1 = pd.DataFrame.as_matrix(otutab_trans)
ratio = round(X1.shape[0]*0.3)
print(ratio)
#ratio = -1
count = 0
for i in xrange(X1.shape[1]):
    if np.count_nonzero(X1[:,i]) > ratio:
    	count = count + 1
print(count)
X2 = np.zeros((X1.shape[0], count))
count = 0
for i in xrange(X1.shape[1]):
    if np.count_nonzero(X1[:,i]) > ratio:
     	X2[:,count] = X1[:,i]
     	count = count + 1
#print(X2)
TrainSample, TestSample, TrainLabel, TestLabel = train_test_split(X2, y, test_size=0.34, random_state=51) #45, 46, 47, 50, 51
#Temp= np.zeros(shape=[len(TrainLabel), n_label], dtype='float32')
print(TestLabel)
prefix = '5'
np.savetxt('../Data/Splits/IT' + prefix + '/TrainSample_' + prefix + '.txt', TrainSample, delimiter='\t')
np.savetxt('../Data/Splits/IT' + prefix + '/TrainLabel_' + prefix + '.txt', TrainLabel, delimiter='\t') 
np.savetxt('../Data/Splits/IT' + prefix + '/TestSample_' + prefix + '.txt', TestSample, delimiter='\t') 
np.savetxt('../Data/Splits/IT' + prefix + '/TestLabel_' + prefix + '.txt', TestLabel, delimiter='\t')  

'''
merge = pd.concat([otutab_trans, metadata[dx]], axis=1,join='inner')


X = merge.drop([dx],axis=1)
y = merge[dx]

n_label = 2
y = y.replace(to_replace=['CD','UC','IC','no'], value=['disorder','disorder','disorder','no-disorder'])
encoder = LabelEncoder()
y = pd.Series(encoder.fit_transform(y), index=y.index, name=y.name)

X1 = pd.DataFrame.as_matrix(X)
ratio = round(X.shape[0]*0.1)
count = 0
for i in xrange(X1.shape[1]):
    if np.count_nonzero(X1[:,i]) > ratio:
    	count = count + 1
print(count)
X2 = np.zeros((X1.shape[0], count))
count = 0
for i in xrange(X1.shape[1]):
    if np.count_nonzero(X1[:,i]) > ratio:
     	X2[:,count] = X1[:,i]
     	count = count + 1
print(X2)

TrainSample, TestSample, TrainLabel, TestLabel = train_test_split(X2, y, test_size=0.34, random_state=47)
Temp= np.zeros(shape=[len(TrainLabel), n_label], dtype='float32')
for i in xrange(len(TrainLabel)):
	Temp[i,TrainLabel[i]] = 1.




np.savetxt('./IBD_new_data/TrainSample_4.txt', TrainSample, delimiter='\t')
np.savetxt('./IBD_new_data/TrainLabel_4.txt', Temp, delimiter='\t') 
np.savetxt('./IBD_new_data/TestSample_4.txt', TestSample, delimiter='\t') 
np.savetxt('./IBD_new_data/TestLabel_4.txt', TestLabel, delimiter='\t')  
'''



