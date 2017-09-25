import scipy.misc
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import NN_functions_param as NNp
import readin as rdin

# Setup parameters
input_layer_size  = 10;  #colors(u-i) *4, color error *5, g*1
hidden_layer_size = 50;   
it_no =100
num_labels = 1;         
reload(NNp)                     

# Load Training Data
print('Loading Data ...\n')

#read data from the files and sort into inputs and labels, training, CV and test sets.
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
sizedf = df.shape[0]/2
sizetrain = sizedf/2
df_train = df[0:sizedf]
df_cv = df[sizedf:sizetrain+sizedf]
df_test = df[sizetrain+sizedf:]

m = sizedf
#choose np.r_[4, 8:18]].values for g_mag, and all errors 
#choose np.r_[3:13, 17]].values for all color and errors
#choose np.r_[3:18]].values for all color and errors
Xt =df_train.iloc[:, np.r_[4, 8:18]].values
X = Xt[:,:-1]
y = Xt[:,-1:]

Xt =df_cv.iloc[:, np.r_[4, 8:18]].values
Xcv = Xt[:,:-1]
ycv = Xt[:,-1:]

Xt =df_test.iloc[:, np.r_[4, 8:18]].values
Xtest = Xt[:,:-1]
ytest = Xt[:,-1:]

print('\nlook for best lambda using CV data \n')
lamparam = 0.5 #turns out to be 0.5

#comment this out after you have best lambda 
""" lambda_vec, error_train, error_val = NNp.validationCurve(X, y, Xcv, ycv, input_layer_size, hidden_layer_size,num_labels, it_no)
error_val, error_train = NNp.learningCurve(X, y, Xcv, ycv, input_layer_size, hidden_layer_size, num_labels,lamparam, it_no)
index_min = np.argmin(error_val)
lamparam = lambda_vec[index_min]
lamparam = 0.02"""

#check if the gradient computation is doing the right thing.
#only for initial debugging, comment out when actually running full code
""" NNp.checkNNGradients(lamparam) #comment out for the training."""

print('\nTrain Neural Network\n')
nn_params = NNp.trainReg(X, y,input_layer_size, hidden_layer_size, num_labels, lamparam, it_no);

# # extract the trained weights Theta1..3 from the long vector nn_params
v1 =(hidden_layer_size * (input_layer_size + 1))
v2 =(hidden_layer_size * (hidden_layer_size + 1))
Theta1 = np.reshape(nn_params[0:v1], (hidden_layer_size, (input_layer_size + 1)))
Theta2 = np.reshape(nn_params[v1:(v1+v2)], (hidden_layer_size, (hidden_layer_size + 1)))
Theta3 = np.reshape(nn_params[(v1+v2):],(num_labels, (hidden_layer_size + 1)))

# Predict whether the objects in the CV and test dataset are QSOs
pred = NNp.predict(Theta1, Theta2, Theta3, Xcv)
predT = NNp.predict(Theta1, Theta2, Theta3, Xtest)
pred_cv = np.copy(pred)
pred_cv[pred<0.5]=0
pred_cv[pred>=0.5]=1
pred_test = np.copy(predT)
pred_test[predT<0.5]=0
pred_test[predT>=0.5]=1

#print out the results on the CV and test set.
print('\nCV Set Accuracy: %f\n', np.mean(pred_cv == ycv) * 100) #just to make sure it makes sense
print('\nTest Set Accuracy: %f\n', np.mean(pred_test == ytest) * 100)
file_out_test = 'testSet_QSOfile_%dit_l%1.2f_Hl%d.txt'  % (it_no, lamparam, hidden_layer_size)
np.savetxt(file_out_test, np.c_[Xtest, ytest, predT], fmt='%1.3f')


