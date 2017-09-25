import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

it_no =1000
lamparam = 0.5
name_plot=['probQSO', 'probPLO']
file_in = 'testSet_QSOfile_%dit_l%1.2f.txt'  % (it_no, lamparam)

#data
df =np.genfromtxt(file_in)
X = df[:,:-1]
y=df[:,-1]

for valnn in range(2):
 
    # masks
    maskPLO = np.zeros(X.shape[0], bool) | (X[:, -1] == 0)
    maskQSO = np.zeros(X.shape[0], bool) | (X[:, -1] == 1)

    #data columns 
    valsx = [6, 7, 8, 9]
    valsy = [7, 8, 9, 6]

    valsxN = ['u-g', 'g-r', 'r-i', 'i-z']
    valsyN = ['g-r', 'r-i', 'i-z', 'u-g']

    fig, axs = plt.subplots(2, 2)
    title_vals = ['QSO NN probabilities','PLO NN probabilities']
    plt.suptitle(title_vals[valnn], fontsize=15)
    params = {'axes.labelsize': 12,'axes.titlesize':12, 'font.size': 12,\
              'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
    plt.rcParams.update(params)
    for i, ax in enumerate(fig.axes):

        #__________________________________________________________"""
        #2D histogram plots_______________________________
        if valnn ==0:
            counts,xbins,ybins,image = ax.hist2d(X[:,valsx[i]][maskQSO], \
            X[:,valsy[i]][maskQSO], (40, 40), alpha=0.0)
            
            count2,xbins,ybins,image = ax.hist2d(X[:,valsx[i]][maskQSO], \
            X[:,valsy[i]][maskQSO], weights= y[maskQSO],bins=(xbins,ybins),alpha=0.0)

            norm_count = np.divide(count2, counts)
            img1 = ax.contourf(norm_count.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
            linewidths=2, cmap = plt.cm.YlGnBu, label='QSO av. prob')

            fig.colorbar(img1, ax=ax,ticks=[0 , 0.5, 1])

        if valnn ==1:
            counts,xbins,ybins,image = ax.hist2d(X[:,valsx[i]][maskPLO], \
            X[:,valsy[i]][maskPLO], (40, 40), alpha=0.0)
            
            count2,xbins,ybins,image = ax.hist2d(X[:,valsx[i]][maskPLO], \
            X[:,valsy[i]][maskPLO], weights= y[maskPLO],bins=(xbins,ybins),alpha=0.0)

            norm_count = np.divide(count2, counts)
            img1 = ax.contourf(norm_count.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
            linewidths=2, cmap = plt.cm.YlGnBu, label='PLO av. prob')

            fig.colorbar(img1, ax=ax,ticks=[0 , 0.5, 1])            
        
        #____________________________________________________
        #plot set up_________________________________________
        ax.legend(loc='upper right')    
        ax.set_ylabel(valsyN[i])
        ax.set_xlabel(valsxN[i])
 
    #______________________________________________________
        
    plt.tight_layout() 
    plt.subplots_adjust(top=0.85)
    plt.show()
    fig.savefig('%s.png'% name_plot[valnn], dpi=600 ) 
  
