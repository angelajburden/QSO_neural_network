import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

it_no =1000
lamparam = 0.5
file_in = 'testSet_QSOfile_%dit_l%1.2f.txt'  % (it_no, lamparam)

#data
df =np.genfromtxt(file_in)
X = df[:,:-1]
y=df[:,-1]
 
# masks
maskPLO = np.zeros(X.shape[0], bool) | (X[:, -1] == 0)
maskQSO = np.zeros(X.shape[0], bool) | (X[:, -1] == 1)
mask_TP = np.zeros(X.shape[0], bool) | ((y>= 0.5) & (X[:, -1] == 1)) #valnn=0 TP
mask_FN = np.zeros(X.shape[0], bool) | ((y<0.5) & (X[:, -1] == 1))   #valnn=1 FN
mask_TN = np.zeros(X.shape[0], bool) | ((y<0.5) & (X[:, -1] == 0))   #valnn=2 TN
mask_FP = np.zeros(X.shape[0], bool) | ((y>=0.5) & (X[:, -1] == 0))  #valnn=3 FP

#bin locations
valx = [0, 1, 2, 3, 4, 5]
    
hvalsx = ['g','error g','error i', 'error r', 'error z', 'error u']
hvalsy ='count'
hname_plot = 'hist_error_log_results'
maxplotx= [22, 0.2, 0.2, 0.2, 0.6, 1.0]

#for 1 D histograms of just yes/no results
"""fig = plt.figure()

plt.hist(y[maskPLO], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         facecolor='b', alpha=0.5, label='PLO' )
plt.hist(y[maskQSO], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         facecolor='r', alpha=0.3, label='QSO' )  
         
plt.hist(y[mask_TP], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='blue', fc='none', lw=1.5, histtype='step', label='TP' )
plt.hist(y[mask_TN], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='red', fc='none', lw=1.5, histtype='step', label='TN' )
plt.hist(y[mask_FP], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='black', fc='none', lw=1.5, histtype='step', label='FP' )
plt.hist(y[mask_FN], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='green', fc='none', lw=1.5, histtype='step', label='FN' )


plt.title("Results of NN on test set") 
plt.legend(loc='upper center') 
plt.ylabel("count")
plt.xlabel("NN classifications")   
# plt.yscale('log')   
plt.xlim([-0.02, 1.02])
plt.ylim([0, 4050])"""
#######NB counts, xedges, yedges, Image)

#for 1D histograms results as a function of input params
params = {'axes.labelsize': 12,'axes.titlesize':12, 'font.size': 12,\
          'legend.fontsize': 8, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
plt.rcParams.update(params)
fig, axs = plt.subplots(3, 2)
suptitle = plt.suptitle('Histograms of NN output for QSO/PLO split by test-sample color error', fontsize=15)

for i, ax in enumerate(fig.axes):
    binsize =(maxplotx[i] - X[:,valx[i]].min())/40.
    
    ax.hist(X[:,valx[i]][mask_TP], 
            bins=np.arange(X[:,valx[i]].min(), maxplotx[i] + binsize, binsize), 
            ec='blue', fc='none', lw=1.5, histtype='step', label='TP' )
    ax.set_yscale('log')       
    ax.hist(X[:,valx[i]][mask_TN], \
         bins=np.arange(X[:,valx[i]].min(), maxplotx[i] + binsize, binsize),
         ec='red', fc='none', lw=1.5, histtype='step', label='TN' )
    ax.set_yscale('log')     
    ax.hist(X[:,valx[i]][mask_FP], \
         bins=np.arange(X[:,valx[i]].min(), maxplotx[i] + binsize, binsize),
         ec='black', fc='none', lw=1.5, histtype='step', label='FP' )
    ax.set_yscale('log')     
    ax.hist(X[:,valx[i]][mask_FN], \
         bins=np.arange(X[:,valx[i]].min(), maxplotx[i] + binsize, binsize),
         ec='green', fc='none', lw=1.5, histtype='step', label='FN' )
    ax.set_yscale('log')   
    ax.legend(loc='upper right')    
    ax.set_ylabel(hvalsy)
    ax.set_xlabel(hvalsx[i])
#     ax.locator_params(nbins=4, axis='x')
#     ax.locator_params(nbins=4, axis='y')   
    
plt.tight_layout() 
plt.subplots_adjust(top=0.85)

fig.savefig('%s.png'% hname_plot, dpi=600)  
plt.show() 
