import scipy.misc
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import readin as rdin
  
#plot info
valsx = ['u-g', 'g-r', 'r-i', 'i-z']
valsy = ['g-r', 'r-i', 'i-z', 'u-g']
dfvalsx = ['u_g', 'g_r', 'r_i', 'i_z']
dfvalsy = ['g_r', 'r_i', 'i_z', 'u_g']

dfhvalsx = ['psfMag_g','psfMagErr_g','psfMagErr_i', 'psfMagErr_r', 'psfMagErr_z', 'psfMagErr_u']
hvalsx = ['g','error g','error i', 'error r', 'error z', 'error u']
hvalsy ='count'
name_plot = 'col_col'
hname_plot = 'hist_cats'
maxplotx= [22, 0.2, 0.2, 0.2, 0.6, 1.0]

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
Xt =df_train.iloc[:, np.r_[4, 8:18]].values
X = Xt[:,:-1]
y = Xt[:,-1:]


maskPLO = df_train['y'] ==0
maskQSO = df_train['y'] ==1

# maskPLOnn = df_train['yNN'] < 0.5
# maskQSOnn = df_train['yNN'] >=0.5

#histograms%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##initial

params = {'axes.labelsize': 12,'axes.titlesize':12, 'font.size': 12,\
          'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
plt.rcParams.update(params)

fig, axs = plt.subplots(3, 2)
plt.suptitle('Histograms of color errors for QSO and PLO in training sample', fontsize=20)

for i, ax in enumerate(fig.axes):
    binsize =(maxplotx[i] - df_train[dfhvalsx[i]].min())/40.
    ax.hist(df_train[dfhvalsx[i]][maskPLO], 
            bins=np.arange(df_train[dfhvalsx[i]].min(), maxplotx[i] + binsize, binsize), 
            facecolor='b', alpha=0.5, label='PLO' )
    ax.hist(df_train[dfhvalsx[i]][maskQSO], 
            bins=np.arange(df_train[dfhvalsx[i]].min(), maxplotx[i] + binsize, binsize), 
            facecolor='r', alpha=0.3, label='QSO' )
    ax.legend(loc='upper right')    
    ax.set_ylabel(hvalsy)
    ax.set_xlabel(hvalsx[i])
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=4, axis='y')   
    
plt.tight_layout() 
plt.subplots_adjust(top=0.85)


# binsize = (maxplotx - df_train[dfvalsx].min())/40
# plt.hist(df_train[dfvalsx][maskPLO], \
#          bins=np.arange(df_train[dfvalsx].min(), maxplotx + binsize, binsize), \
#          facecolor='b', alpha=0.5, label='PLO' )
# plt.hist(df_train[dfvalsx][maskQSO], \
#          bins=np.arange(df_train[dfvalsx].min(), maxplotx + binsize, binsize), \
#          facecolor='r', alpha=0.3, label='QSO' )

##results
# plt.hist(df_train['yNN'][maskPLO], \
#          bins=np.arange(df_train['yNN'].min(), maxplotx + binsize, binsize), \
#          facecolor='b', alpha=0.5, label='PLO' )
# plt.hist(df_train['yNN'][maskQSO], \
#          bins=np.arange(df_train['yNN'].min(), maxplotx + binsize, binsize), \
#          facecolor='r', alpha=0.3, label='QSO' )
# plt.locator_params(axis='y', nticks=6)
# plt.locator_params(axis='x', nticks=4) 
      
#scatterplots%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

##initial

# params = {'axes.labelsize': 12,'axes.titlesize':12, 'font.size': 12,\
#           'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
# plt.rcParams.update(params)
# 
# fig, axs = plt.subplots(2, 2)
# plt.suptitle('Color distributions of QSO and PLO in training sample', fontsize=20)
# 
# 
# for i, ax in enumerate(fig.axes):
#     ax.scatter(df_train[dfvalsx[i]][maskPLO], 
#             df_train[dfvalsy[i]][maskPLO], color='b', alpha=0.7, s = 30, label='PLO')
#     ax.scatter(df_train[dfvalsx[i]][maskQSO], 
#             df_train[dfvalsy[i]][maskQSO], color='r', alpha=0.2, s = 30, label='QSO')
#     ax.legend(loc='upper right')    
#     ax.set_ylabel(valsy[i])
#     ax.set_xlabel(valsx[i])
#     ax.locator_params(nbins=4, axis='x')
#     ax.locator_params(nbins=4, axis='y')   
#     
# plt.tight_layout() 
# plt.subplots_adjust(top=0.85)
       
##results           
# plt.scatter(df_train[dfvalsx][maskPLO], 
#             df_train[dfvalsy][maskPLO], marker='o', facecolors='none', edgecolors='k', alpha=1, s = 124, label='PLO')
# plt.scatter(df_train[dfvalsx][maskQSO], 
#             df_train[dfvalsy][maskQSO], marker='s', facecolors='none', edgecolors='g', alpha=1, s = 124, label='QSO')
# plt.scatter(df_train[dfvalsx][maskQSOnn], 
#             df_train[dfvalsy][maskQSOnn], marker='*', color='#a8ddb5', alpha=1.0, s = 124, label='QSOnn')
# plt.xlim([0-0.5, 4+0.5])
# plt.ylim([0-0.5, 4+0.5])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
plt.show()
fig.savefig('%s.jpg'% hname_plot )
