from sklearn.neural_network import MLPClassifier
import scipy.misc
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import NN_functions_param as NNp
import readin as rdin

# Setup parameters
input_layer_size  = 10;  #colors(u-i) *4, color error *5, g*1
hidden_layer_size = 200;   
num_labels = 1;         
reload(NNp)                     

# Load Training Data
print('Loading Data ...\n')
#read data from the files and sort into inputs and labels, training, test sets.
QSO_file = 'QSO.csv'
BG_file = 'BG_STAR.csv'
df1 = pd.read_csv(QSO_file)
df2 = pd.read_csv(BG_file)
#apply cuts
df1 = rdin.df_masks(df1)
df2 = rdin.df_masks(df2)
df1 = df1.assign(y=1)
df2 = df2.assign(y=0)

result = pd.concat([df1, df2], ignore_index=True)
df = result.reindex(np.random.permutation(result.index))
del df1
del df2
sizedf = df.shape[0]*3/4
df_train = df[0:sizedf]
df_test = df[sizedf:]
m = sizedf

Xt =df_train.iloc[:, np.r_[4, 8:18]].values
X = Xt[:,:-1]
y = Xt[:,-1:]
print(X.shape, y.shape)

Xt =df_test.iloc[:, np.r_[4, 8:18]].values
X_test = Xt[:,:-1]
y_test = Xt[:,-1:]

clf = MLPClassifier(activation='logistic',solver='lbfgs', alpha=0.02, hidden_layer_sizes=(hidden_layer_size, 2),verbose=True, random_state=1)
clf.fit(X, y)
b = clf.predict(X_test)
prob = clf.predict_proba(X_test)
b = np.reshape(b, [len(b),1])
c = y_test==b
score = float(sum(c))/len(b)

print(score)