#===================================================This script contains codes for ===================================================
#Direct prediction of bioaccumulation of organic contaminants in plant roots from soils with machine learning models based on molecular structures

import pandas as pd
import numpy as np
import scipy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


print('To generate and save figures automatically, please uncomment savefig codes. reading data')
data = pd.read_csv('../Data/rcf_lipid_combined.csv',encoding='cp1252')
print('finish reading data')



MW = data['MW'].to_numpy().reshape(-1,1)
KOW = data['log Kow'].to_numpy().reshape(-1,1)
OM = data['fom (%)'].to_numpy().reshape(-1,1)
RCF = data['log RCF-soil'].to_numpy()
SMILES = data['SMILES'].to_numpy()
FLIPID = data['flip (%)'].to_numpy()
COMPOUNDS = data['ï»¿Compounds'].to_numpy()


FP = []
FP_ = []
NAME = []
MW_unique = []
logKow_unqiue = []
for i,sm in enumerate(SMILES):
    mol = Chem.MolFromSmiles(sm)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    if np.array(fp).tolist() not in FP:
        FP.append(np.array(fp).tolist())
        FP_.append(fp)
        NAME.append(COMPOUNDS[i])
        MW_unique.append(MW[i])
        logKow_unqiue.append(KOW[i])



#=============run the following codes to visualize basic description of the dataset in the main paper=====================


plt.figure(figsize=(7,7))
plt.subplots_adjust(wspace=1.5)
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

#display of molecular weight of all unique chemicals in this dataset
sns.boxplot(np.array(MW_unique).reshape(-1),ax=ax1,orient='v',color='red')
sns.swarmplot(np.array(MW_unique).reshape(-1),ax=ax1,orient='v',color="red",s=3.5)        
ax1.set_title('$MW$')

#display of logKow values of all unique chemicals in this dataset
sns.boxplot(np.array(logKow_unqiue).reshape(-1),ax=ax2,orient='v',color='green')
sns.swarmplot(np.array(logKow_unqiue).reshape(-1),ax=ax2,orient='v',color="green",s=3.5) 
ax2.set_title('$logK_{ow}$')

#display of unqiue f_om values in this dataset
sns.boxplot(OM,ax=ax3,orient='v',color='yellow')
sns.swarmplot(list(set(OM.reshape(-1))),ax=ax3,orient='v',color="lightyellow",s=3.5) 
ax3.set_title('$f_{OM}/\%$')

#display of unique f_lipid values in this dataset
sns.boxplot(list(set(FLIPID.reshape(-1))),ax=ax4,orient='v',color='blue')
sns.swarmplot(list(set(FLIPID.reshape(-1))),ax=ax4,orient='v',color="blue",s=3.5) 
ax4.set_title('$f_{lipid}/\%$')

#display of unique logRCF values in this dataset
sns.boxplot(list(set(RCF.reshape(-1))),ax=ax5,orient='v',color='orange')
sns.swarmplot(list(set(RCF.reshape(-1))),ax=ax5,orient='v',color="orange",s=3.5) 
ax5.set_title('$logRCF_{soil}$')

#display of all logRCF values in this dataset; can be used to replace the unique one
#sns.boxplot(list(RCF.reshape(-1)),ax=ax5,orient='v',color='orange')
#sns.swarmplot(list(RCF.reshape(-1)),ax=ax5,orient='v',color="orange",s=3.5) 
#ax5.set_title('$logRCF_{soil}$')

for axi in [ax1,ax2,ax3,ax4,ax5]:
    for i,patch in enumerate(axi.artists):
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))
        patch.set_edgecolor('black')

        for j in range(6*i,6*(i+1)):
             axi.lines[j].set_color('black')
                
#save the figure to current folder
#plt.savefig('data_stat',dpi=600)


#=============calculation of similarity matrix using dice similarity based on ECFP=========================



similarity_matrix = np.zeros((len(NAME),len(NAME)))
for i,fp1 in enumerate(FP_):
    for j in range(i,len(FP_)):
        similarity_matrix[i][j] = DataStructs.DiceSimilarity(fp1,FP_[j])
        similarity_matrix[j][i] = DataStructs.DiceSimilarity(fp1,FP_[j])


#=================run the following codes to generate heatmap of the similarity matrix=======================


print('start generating heatmap')



mask = np.triu(np.ones_like(similarity_matrix, dtype=np.bool))
#np.fill_diagonal(mask,False)
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(similarity_matrix, mask=mask, cmap=cmap, vmax=1., center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})#,xticklabels=NAME,yticklabels = NAME)

plt.xticks(fontsize=7, rotation=90)
plt.yticks(fontsize=7)
#plt.savefig('heatmap',dpi=600)


#=============run below codes for clustering chemicals based on ECFP using kmeans algorithum===============



print('start clustering chemical strucutre')



from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(FP)
pca_2d = pca.transform(FP)


from sklearn.cluster import KMeans
k_means = KMeans(random_state=1, n_clusters=4)
k_means.fit(pca_2d)
cluster = k_means.predict(pca_2d)


sns.scatterplot(pca_2d[:,0],pca_2d[:,1],hue=cluster,palette={3:"#C10A36",2:"#3E7E6C",0:"#EECF63",1:'#6D042C'})
plt.xlabel('PCA1')
plt.ylabel('PCA2')
#plt.savefig('clustering',dpi=600)


#================run codes below to assign chemical names to different groups=================


g0 = []
g1 = []
g2 = []
g3 = []

for i,name in enumerate(NAME):
    if cluster[i] == 0:
        g0.append(name)
    elif cluster[i] == 1:
        g1.append(name)
    elif cluster[i] == 2:
        g2.append(name)
    else:
        g3.append(name)
print('group0 contains',g0)
print('group1 contains',g1)
print('group2 contains',g2)
print('group3 contains',g3)
