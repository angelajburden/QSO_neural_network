import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

it_no =1000
lamparam = 0.5
name_plot=['scatter_TP','scatter_FN','scatter_TN','scatter_FP']
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
valsx = [6, 7, 8, 9]
valsy = [7, 8, 9, 6]

valsxN = ['u-g', 'g-r', 'r-i', 'i-z']
valsyN = ['g-r', 'r-i', 'i-z', 'u-g']
valnn=2
fig, axs = plt.subplots(2, 2)
title_vals = ['NN probabilities TP','NN probabilities FN','NN probabilities TN','NN probabilities FP']
plt.suptitle(title_vals[valnn], fontsize=15)
params = {'axes.labelsize': 12,'axes.titlesize':12, 'font.size': 12,\
              'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
plt.rcParams.update(params)
for i, ax in enumerate(fig.axes):

        #scatter plots________________________________________________
#        ax.scatter(X[:,valsx[i]][maskPLO], 
#             X[:,valsy[i]][maskPLO],c=y marker='.', facecolors='none', edgecolors='b', alpha=1.0, s = 60, label='PLO')  
                      
#        ax.scatter(X[:,valsx[i]][maskQSO], 
#             X[:,valsy[i]][maskQSO], marker='.', color='r',  alpha=1, s = 60, label='QSO') #facecolors='none', edgecolors='r',
                        
    img1 =ax.scatter(X[:,valsx[i]][mask_FP], 
            X[:,valsy[i]][mask_FP], c=y[mask_FP], marker='o', edgecolor='none', cmap=plt.cm.YlGnBu, alpha=1.0, s = 20) 
                         
#         ax.scatter(X[:,valsx[i]][mask_FN], 
#             X[:,valsy[i]][mask_FN], marker='o', color='r', alpha=0.3, s = 20, label='False -') 
#                        
#         ax.scatter(X[:,valsx[i]][mask_TN], 
#             X[:,valsy[i]][mask_TN], marker='.', color='b', alpha=0.3, s = 10, label='True -')  
#                         
#         ax.scatter(X[:,valsx[i]][mask_FP], 
#             X[:,valsy[i]][mask_FP], marker='o', color='g', alpha=0.3, s = 20, label='False +')
        #__________________________________________________________
    cbar =fig.colorbar(img1, ax=ax)
    cbar.set_ticks([0.6,0.9])
    cbar.set_ticklabels(['0.6','0.9'])
#     cbar.ax.set_yticklabels(['0.5', '1'])
        #____________________________________________________
        #plot set up_________________________________________
    ax.legend(loc='upper right')    
    ax.set_ylabel(valsyN[i])
    ax.set_xlabel(valsxN[i])
#     ax.locator_params(nbins=4, axis='x')
#     ax.locator_params(nbins=4, axis='y')   
#     ax.set_xlim([-1, 4])
#     ax.set_ylim([-1, 4])
    #______________________________________________________
        
plt.tight_layout() 
plt.subplots_adjust(top=0.85)
plt.show()
fig.savefig('%s.png'% name_plot[valnn], dpi=600 ) 