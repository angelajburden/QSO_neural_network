####read - this doesn't seem to work, only goes through 1 iteration and marks all as PLO
#####accuracy looks good as the sample is 92% PLO BUT none of the QSOs are recognised

import scipy.misc
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import NN_functions_param as NNp
# import readin as rdin

# Setup parameters
input_layer_size  = 10;  #colors(u-i) *4, color error *5, g*1
hidden_layer_size = 50; 
  
itf_no =1000
it_no =200
num_labels = 1; 
lamparam = 0.5        
reload(NNp)    

# Load Training Data
print('Loading Data ...\n')                
file1 = 'results/cValidation_QSOfile_%dit_l%1.2f.txt'  % (itf_no, lamparam)
file2 = 'results/testSet_QSOfile_%dit_l%1.2f.txt'  % (itf_no, lamparam)
#data
df =np.genfromtxt(file1)

#we want the qso that have y=0, X[:,-1]=1, i.e. the ones we missed
# masks
maskPLO = np.zeros(df.shape[0], bool) | (df[:, -2] == 0)
maskQSO = np.zeros(df.shape[0], bool) | (df[:, -2] == 1)
mask_FN = np.zeros(df.shape[0], bool) | ((df[:,-1]<0.5) & (df[:, -2] == 1))#FN
Xmissed = df[mask_FN]
XPLO    = df[maskPLO]

#combine them
Xtemp = np.concatenate((Xmissed, XPLO), axis=0)
np.random.shuffle(Xtemp)
print(Xtemp.shape)

X =Xtemp[:,:-2]
y =Xtemp[:,-2] 

print('\nTrain Neural Network\n')
nn_params = NNp.trainReg(X, y,input_layer_size, hidden_layer_size, num_labels, lamparam, it_no);

# # extract the trained weights Theta1..3 from the long vector nn_params
v1 =(hidden_layer_size * (input_layer_size + 1))
v2 =(hidden_layer_size * (hidden_layer_size + 1))
Theta1 = np.reshape(nn_params[0:v1], (hidden_layer_size, (input_layer_size + 1)))
Theta2 = np.reshape(nn_params[v1:(v1+v2)], (hidden_layer_size, (hidden_layer_size + 1)))
Theta3 = np.reshape(nn_params[(v1+v2):],(num_labels, (hidden_layer_size + 1)))

# Predict whether the objects in the CV and test dataset are QSOs

#same for test set (should be function do later)
df =np.genfromtxt(file2)
# masks
maskPLO = np.zeros(df.shape[0], bool) | (df[:, -2] == 0)
maskQSO = np.zeros(df.shape[0], bool) | (df[:, -2] == 1)
mask_FN = np.zeros(df.shape[0], bool) | ((df[:,-1]<0.5) & (df[:, -2] == 1))#FN
Xmissed = df[mask_FN]
XPLO = df[maskPLO]
#combine them
Xtemp = np.concatenate((Xmissed, XPLO), axis=0)
np.random.shuffle(Xtemp)
# Xt3 = Xt2.reindex(np.random.permutation(Xt2.index))
Xtest =Xtemp[:,:-2]
ytest =Xtemp[:,-2] 

# pred = NNp.predict(Theta1, Theta2, Theta3, Xcv)
predT = NNp.predict(Theta1, Theta2, Theta3, Xtest)
# pred_cv = np.copy(pred)
# pred_cv[pred<0.5]=0
# pred_cv[pred>=0.5]=1
pred_test = np.copy(predT)
pred_test[predT<0.5]=0
pred_test[predT>=0.5]=1

#print out the results on the CV and test set.
# print('\nCV Set Accuracy: %f\n', np.mean(pred_cv == ycv) * 100) #just to make sure it makes sense
print('\nTest Set Accuracy: %f\n', np.mean(pred_test == ytest) * 100)
file_out_test = 'testSet_QSOfile_%dit_l%1.2f_Hl%d_secondNN.txt'  % (it_no, lamparam, hidden_layer_size)
np.savetxt(file_out_test, np.c_[Xtest, ytest, predT], fmt='%1.3f')


