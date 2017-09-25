import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

name_plot = 'params_NN'
its=[10,50,100,200,500,1000]
CFv=[0.561,0.453,0.369,0.259,0.215,0.207]
acc = [78.13,82.01,89.28,90.81,92.78,92.84]

lambdav = [0.02, 0.5, 1, 2, 3, 5] #2
lamcost = [0.401, 0.396, 0.377, 0.328, 0.417]
lamacc  = [87.55, 89.28, 87.54, 89.32, 88.33, 87.10]

hlno = [20, 50, 80, 100]
hlcost=[0.344, 0.369, 0.410, 0.396]
hlacc=[87.30, 89.28, 87.75, 85.39]

params = {'axes.labelsize': 14,'axes.titlesize':14, 'font.size': 14,\
          'legend.fontsize': 8, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
plt.rcParams.update(params)
fig, axs = plt.subplots(2, 2)
suptitle = plt.suptitle('Parameters for NN', fontsize=15)

params = {'axes.labelsize': 14,'axes.titlesize':14, 'font.size': 14,\
          'legend.fontsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
plt.rcParams.update(params)
    
# axs[0][0].plot(CFv,acc,'r')  #0.2-0.6
# axs[0][0].scatter(CFv,acc,c='k')  #0.2-0.6
axs[0][1].plot(its, CFv,'g')
axs[0][1].scatter(its, CFv,c='k')

axs[1][1].plot(its,acc, 'b')
axs[1][1].scatter(its,acc, c='k')
  
axs[0][0].plot(lambdav,lamacc,'r')  #0.2-0.6
axs[0][0].scatter(lambdav,lamacc,c='k')  #0.2-0.6
axs[1][0].plot(hlno, hlacc,'g')
axs[1][0].scatter(hlno, hlacc,c='k')

# axs[0][0].set_xlim([0.18, 0.62])
# axs[0][0].set_xticks([0.2, 0.4, 0.6])
# axs[0][0].set_yticks([75, 90, 100])
# axs[0][0].set_ylabel('Accuracy [%]')
# axs[0][0].set_xlabel('Cost')

axs[0][1].set_xlim([0, 1050])
axs[0][1].set_xticks([10, 200, 500, 1000])
axs[0][1].set_ylabel('Cost')
axs[0][1].set_xlabel('Iterations')

axs[1][1].set_xlim([0, 1050])
axs[1][1].set_xticks([10, 200, 500, 1000])
axs[1][1].set_yticks([75, 90, 100])
axs[1][1].set_ylabel('Accuracy [%]')
axs[1][1].set_xlabel('Iterations')

axs[0][0].set_xticks([0, 1, 5])
axs[0][0].set_yticks([85, 90])
axs[0][0].set_xlabel('Lambda value')
axs[0][0].set_ylabel('Accuracy [%]')

axs[1][0].set_xticks([20, 50, 100])
axs[1][0].set_yticks([85, 90])
axs[1][0].set_xlabel('HL nodes')
axs[1][0].set_ylabel('Accuracy [%]')

axs[0][0].annotate('200 it, 50 HLNs', xy=(0, 86))
axs[1][0].annotate('200 it, lambda =0.5', xy=(20, 86))
axs[0][1].annotate('lambda =0.5, 50 HLNs', xy=(200, 0.5))
axs[1][1].annotate('lambda =0.5, 50 HLNs', xy=(200, 78))

plt.tight_layout() 
plt.subplots_adjust(top=0.85)
plt.show()
 
fig.savefig('%s.png'% name_plot, dpi=600)  
 
