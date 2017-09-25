import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

it_no =1000
lamparam = 0.5
name_plot=['contour_ratio_QSO','contour_ratio_missedQSO','contour_ratio PLO','contour_ratio_missedPLO']
file_in = 'testSet_QSOfile_%dit_l%1.2f.txt'  % (it_no, lamparam)

#data
df =np.genfromtxt(file_in)
X = df[:,:-1]
y=df[:,-1]

for valnn in range(4):
 
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

    fig, axs = plt.subplots(2, 2)
    title_vals = ['Fraction of QSO caught by NN','Fraction of QSO missed by NN','Fraction of PLO caught by NN','Fraction of PLO missed by NN']
    plt.suptitle(title_vals[valnn], fontsize=15)
    params = {'axes.labelsize': 12,'axes.titlesize':12, 'font.size': 12,\
              'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
    plt.rcParams.update(params)
    for i, ax in enumerate(fig.axes):

        #__________________________________________________________"""
        #2D histogram plots_______________________________
        #counts,ybins,xbins,image = ax.hist2d(X[:,valsx[i]][maskQSO], X[:,valsy[i]][maskQSO], (20, 20),norm=LogNorm(), cmap=plt.cm.Greys, label='QSO'  )
        if valnn < 2:
            counts,xbins,ybins,image = ax.hist2d(X[:,valsx[i]][maskQSO], \
            X[:,valsy[i]][maskQSO], (80, 80), norm=LogNorm(),cmap=plt.cm.BuPu, alpha=0, label='QSO')
        else:
            counts,xbins,ybins,image = ax.hist2d(X[:,valsx[i]][maskPLO], \
            X[:,valsy[i]][maskPLO], (80, 80), norm=LogNorm(),cmap=plt.cm.YlOrRd, alpha=0, label='PLO')

        if valnn == 0:
            counts2,xbins2,ybins2,image2 = ax.hist2d(X[:,valsx[i]][mask_TP],\
            X[:,valsy[i]][mask_TP], bins=(xbins, ybins),norm=LogNorm(),alpha=0) 
            
            countR = np.divide(counts2, counts)
            img1 =ax.contourf(countR.transpose(),extent=[xbins2[0],xbins2[-1],ybins2[0],ybins2[-1]],
            linewidths=2, cmap = plt.cm.YlGnBu, label='QSO, TP')
            
            fig.colorbar(img1, ax=ax,ticks=[0 , 0.5, 1])
            
        if valnn == 1:
            counts2,xbins2,ybins2,image2 = ax.hist2d(X[:,valsx[i]][mask_FN],\
            X[:,valsy[i]][mask_FN], bins=(xbins, ybins),norm=LogNorm(),alpha=0)
             
            countR = np.divide(counts2, counts)
            img1 = ax.contourf(countR.transpose(),extent=[xbins2[0],xbins2[-1],ybins2[0],ybins2[-1]],
            linewidths=2, cmap = plt.cm.YlGnBu, label='QSO, FN')
  
            fig.colorbar(img1, ax=ax,ticks=[0 , 0.5, 1])
            
        if valnn == 2:
            counts2,xbins2,ybins2,image2 = ax.hist2d(X[:,valsx[i]][mask_TN],\
            X[:,valsy[i]][mask_TN], bins=[xbins, ybins],norm=LogNorm(),alpha=0) 
            
            countR = np.divide(counts2, counts)
            img1 = ax.contourf(countR.transpose(),extent=[xbins2[0],xbins2[-1],ybins2[0],ybins2[-1]],
            linewidths=2, cmap = plt.cm.YlGnBu, label='PLO, TN')

            fig.colorbar(img1, ax=ax,ticks=[0 , 0.5, 1])
        if valnn == 3:
            counts2,xbins2,ybins2,image2 = ax.hist2d(X[:,valsx[i]][mask_FP],\
            X[:,valsy[i]][mask_FP], bins=[xbins, ybins],norm=LogNorm(),alpha=0) 
            
            countR = np.divide(counts2, counts)
            img1 = ax.contourf(countR.transpose(),extent=[xbins2[0],xbins2[-1],ybins2[0],ybins2[-1]],
            linewidths=2, cmap = plt.cm.YlGnBu, label='PLO, FP')       
        
            fig.colorbar(img1, ax=ax,ticks=[0 , 0.5, 1])
        #____________________________________________________
        #plot set up_________________________________________
        ax.legend(loc='upper right')    
        ax.set_ylabel(valsyN[i])
        ax.set_xlabel(valsxN[i])
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=4, axis='y')   
#     ax.set_xlim([-1, 4])
#     ax.set_ylim([-1, 4])
    #______________________________________________________
        
    plt.tight_layout() 
    plt.subplots_adjust(top=0.85)
#     plt.show()
    fig.savefig('%s.png'% name_plot[valnn], dpi=600 ) 
  
