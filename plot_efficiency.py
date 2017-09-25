import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

it_no =1000
lamparam = 0.5
file_in = 'results/testSet_QSOfile_%dit_l%1.2f.txt'  % (it_no, lamparam)
name_plot = 'efficiency_plot'
#data
df =np.genfromtxt(file_in)
X = df[:,:-1]
y=df[:,-1]
 
# masks
maskPLO = np.zeros(X.shape[0], bool) | (X[:, -1] == 0)
maskQSO = np.zeros(X.shape[0], bool) | (X[:, -1] == 1)

effcy = [0.2, 0.5, 0.8, 0.9, 0.95, 0.98]
labels = ['0.2','0.5', '0.8', '0.9', '0.95', '0.98']
no_QSO = sum(maskQSO)
no_PLO = sum(maskPLO)

yNN_QSO = y[maskQSO]
yNN_PLO = y[maskPLO]
xeff=np.zeros(len(effcy))
yeff=np.zeros(len(effcy))
xeff2=np.zeros(len(range(100)))
yeff2=np.zeros(len(range(100)))
del_ef = 1.0/100

for j in range(100):
    xeff2[j] =float(sum(yNN_QSO > (j*del_ef)))/no_QSO
    yeff2[j] =float(sum(yNN_PLO > (j*del_ef)))/no_PLO   
for i in range(6):
    xeff[i] =float(sum(yNN_QSO > effcy[i]))/no_QSO
    yeff[i] =float(sum(yNN_PLO > effcy[i]))/no_PLO

params = {'axes.labelsize': 18,'axes.titlesize':18, 'font.size': 18,\
          'legend.fontsize': 14, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
plt.rcParams.update(params)
    
plt.yscale('log') 
# plt.xscale('log')
plt.plot(xeff2, yeff2, color='r')    
plt.scatter(xeff,yeff, c='k')

plt.ylabel("PLO efficiency")
plt.xlabel("QSO efficiency")   
  
plt.xlim([0.4, 1.00])
plt.ylim([0.001, 1.00])
plt.title("Efficiency") 
plt.grid(linestyle='--')
for label, x, y in zip(labels,xeff,yeff):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.show()
 
# plt.savefig('%s.png'% name_plot, dpi=600)  
 
