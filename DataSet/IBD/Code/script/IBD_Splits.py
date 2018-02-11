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
import seaborn

# Variables to call
otuinfile = './dgerver_cd_otu.txt'
mapfile = './mapfile_dgerver.txt'
# Column name from mapfile for disease
dx = 'gastrointest_disord'

# Data read
data = pd.read_table(otuinfile,sep='\t',index_col=0,skiprows=1)
metadata = pd.read_table(mapfile,sep='\t',index_col=0)
#print data

otutab_trans = data.transpose()
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




