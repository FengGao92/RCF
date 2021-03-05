#====================================================This script contains codes for ===================================================
#Direct prediction of bioaccumulation of organic contaminants in plant roots from soils with machine learning models based on molecular structures

import pandas as pd
import numpy as np
import scipy
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence


from rdkit import Chem
from rdkit.Chem import AllChem
from utility import Kfold

#=================run below codes to preprocess data=========================


print('reading data')
data = pd.read_csv('../Data/rcf_lipid_combined.csv',encoding='cp1252')
print('reading data finished')


SMILES = data['SMILES'].to_numpy()
MW = data['MW'].to_numpy().reshape(-1,1)
logKOW = data['log Kow'].to_numpy().reshape(-1,1)
OM = data['fom (%)'].to_numpy().reshape(-1,1)
flipid = data['flip (%)'].to_numpy().reshape(-1,1)

#RCF_water = data['log RCF- water'].to_numpy()
RCF_soil = data['log RCF-soil'].to_numpy()

#generate ECFP for all molecules
print('statr generating ECFP')
FP = []
ONS_index = []
for i,sm in enumerate(SMILES):
    if 'N' in sm:#or 'O' in sm or 'S' in sm:
        ONS_index.append(i)
    mol = Chem.MolFromSmiles(sm)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    FP.append(fp)
FP = np.array(FP)
print('finish generating ECFP')


n_sample = len(RCF_soil)


#====================run below codes for GBRT model with physicochemical properties


print('start 5-fold cross-validation for GBRT-property model')
feature_wo_smiles = np.concatenate((MW,logKOW,OM,flipid),1)
feature_wo_smiles_z = scipy.stats.mstats.zscore(feature_wo_smiles,0)
#uncomment and run below codes to shuffle the data, or directly load sample index to do test run
#total_id = np.arange(n_sample)
#np.random.shuffle(total_id)
total_id = np.load('sample_index.npy')
splits = 5
train_split_index,test_split_index = Kfold(n_sample,splits)

prediction = []
prediction_true = []
test_score_all = []
feature_importance_all = []

importance_all_dots = []
feature_importance_permute = []
for k in range(splits):
    
    print('split is ',k)
    train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
    valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
    test_index = test_split_index[k]
    
    train_id = [total_id[i] for i in train_index]
    valid_id = [total_id[i] for i in valid_index]
    test_id = [total_id[i] for i in test_index]
    train_feature = [feature_wo_smiles_z[i] for i in train_id]
    train_label = [RCF_soil[i] for i in train_id]
    
    valid_feature = [feature_wo_smiles_z[i] for i in valid_id]
    valid_label = [RCF_soil[i] for i in valid_id]
    
    test_feature = [feature_wo_smiles_z[i] for i in test_id]
    test_label = [RCF_soil[i] for i in test_id]
    
    n_estimator = [200,250,500,750,1000,1250]
    max_depths = [2,4,6,8,10]
    
    best_valid_score = 0
    for ne in n_estimator:
        for m_d in max_depths:
            model = GradientBoostingRegressor(n_estimators=ne,max_depth=m_d,learning_rate=0.1)
            model.fit(np.array(train_feature),np.array(train_label).reshape(-1))
            valid_score = model.score(valid_feature,np.array(valid_label).reshape(-1,1))
            if valid_score>best_valid_score:
                best_valid_score = valid_score
                test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
                pred = model.predict(test_feature)
                best_n = ne
                best_d = m_d
                
    model = GradientBoostingRegressor(n_estimators=best_n,max_depth=best_d).fit(np.array(train_feature),np.array(train_label))
    permut_importance = permutation_importance(model,test_feature,np.array(test_label),n_repeats=10)
    importance_all_dots.append(permut_importance.importances)
    
    plot_partial_dependence(model, np.array(train_feature)[:,0].reshape(-1,1),['$MW$'],np.array(['$MW$']))
    #uncomment code below to output and save figure
    #plt.savefig('MW_pdp'+str(k),dpi=600)
    plot_partial_dependence(model, np.array(train_feature)[:,1].reshape(-1,1),['$logKow$'],np.array(['$logKow$']))
    #plt.savefig('Kow_pdp'+str(k),dpi=600)
    plot_partial_dependence(model, np.array(train_feature)[:,2].reshape(-1,1),['$f_{OM}$'],np.array(['$f_{OM}$']))
    #plt.savefig('OM_pdp'+str(k),dpi=600)
    plot_partial_dependence(model, np.array(train_feature)[:,3].reshape(-1,1),['$f_{lipid}$'],np.array(['$f_{lipid}$']))
    #plt.savefig('lipid_pdp'+str(k),dpi=600)
    
    print('test score for this split',test_score)
    prediction.append(pred)
    prediction_true.append(test_label)
    test_score_all.append(test_score)
    feature_importance_all.append(model.feature_importances_)
    
    feature_importance_permute.append(permut_importance.importances_mean)
    print('best n_estimator is',best_n)
    print('best depth is',best_d)
    print('feature importance',model.feature_importances_)


#read in chemical group obtained from clustering results
ccmap = np.load('chemical_group.npy')


ccmap_shuffle_property = []
for i in total_id:
    ccmap_shuffle_property.append(ccmap[i])

prediction_wo_smiles = []
for l in prediction:
    for v in l:
        prediction_wo_smiles.append(v)


prediction_true_wo_smiles = []
for l in prediction_true:
    for v in l:
        prediction_true_wo_smiles.append(v)


sns.scatterplot(prediction_true_wo_smiles,prediction_wo_smiles,linewidth=0.5,hue=ccmap_shuffle_property,palette={1:'#6D042C',0:"#EECF63",2:"#3E7E6C",3:"#C10A36"},s=35)
sns.lineplot(np.arange(-3.,2.0),np.arange(-3,2.0),color='r')
plt.xlabel('Measured $logRCF_{soil}$')
plt.ylabel('Predicted $logRCF_{soil}$')
plt.text(-3.1,1,"R-squared = %0.2f" % np.mean(test_score_all),ha='left',va='top')
plt.legend(loc='lower right')
#uncomment code below to ouput and save figure, there may be slightly difference due to random state and/or package versions
#plt.savefig('GBRT-property_r2',dpi=600)


print('test r2 for GBRT-property model is', np.mean(test_score_all))
print('GBRT-property Finished')


#=======================feature w/ ECFP======================
print('start 5-fold cross-validation for GBRT-ECFP model')
feature_w_smiles = np.concatenate((FP,OM,flipid),1)
feature_w_smiles_z = scipy.stats.mstats.zscore(feature_w_smiles,0)
feature_w_smiles_z[np.isnan(feature_w_smiles_z)] = 0

train_split_index,test_split_index = Kfold(n_sample,5)

splits = 5
prediction_w_smiles = []
prediction_true_w_smiles = []
test_score_all_w_smiles = []
feature_importance_all_w_smiles = []

importance_all_dots_w_smiles_test = []
permute_importance_all_w_smiles_test = []


for k in range(splits):
    
    print('split is ',k)
    train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
    valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
    test_index = test_split_index[k]
    
    train_id = [total_id[i] for i in train_index]
    valid_id = [total_id[i] for i in valid_index]
    test_id = [total_id[i] for i in test_index]
    
    train_feature = [feature_w_smiles_z[i] for i in train_id]
    train_label = [RCF_soil[i] for i in train_id]
    
    valid_feature = [feature_w_smiles_z[i] for i in valid_id]
    valid_label = [RCF_soil[i] for i in valid_id]
    
    test_feature = [feature_w_smiles_z[i] for i in test_id]
    test_label = [RCF_soil[i] for i in test_id]
    
    n_estimator = [200,250,500,750,1000,1250]
    max_depths = [2,4,6,8,10]
    
    best_valid_score = 0
    for ne in n_estimator:
        for m_d in max_depths:
            model = GradientBoostingRegressor(n_estimators=ne,max_depth=m_d,learning_rate=0.1)
            model.fit(np.array(train_feature),np.array(train_label).reshape(-1))
            valid_score = model.score(valid_feature,np.array(valid_label).reshape(-1,1))
            if valid_score>best_valid_score:
                best_valid_score = valid_score
                test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))                
                pred = model.predict(test_feature)
                best_n = ne
                best_d = m_d
    
    model = GradientBoostingRegressor(n_estimators=best_n,max_depth=best_d).fit(np.array(train_feature),np.array(train_label))
    
    permut_importance_test = permutation_importance(model,test_feature,np.array(test_label),n_repeats=10)
    importance_all_dots_w_smiles_test.append(permut_importance_test.importances)
    
    print('test score for this split',test_score)
    prediction_w_smiles.append(pred)
    prediction_true_w_smiles.append(test_label)
    test_score_all_w_smiles.append(test_score)
    feature_importance_all_w_smiles.append(model.feature_importances_)
    
    permute_importance_all_w_smiles_test.append(permut_importance_test.importances_mean)
    print('best n_estimator is',best_n)
    print('best depth is',best_d)
    print('feature importance',model.feature_importances_)



prediction_w_smiles_ = []
for l in prediction_w_smiles:
    for v in l:
        prediction_w_smiles_.append(v)



prediction_true_w_smiles_ = []
for l in prediction_true_w_smiles:
    for v in l:
        prediction_true_w_smiles_.append(v)



ccmap_shuffle_smiles = []
for i in total_id:
    ccmap_shuffle_smiles.append(ccmap[i])



sns.scatterplot(prediction_true_w_smiles_,prediction_w_smiles_,linewidth=0.5,hue=ccmap_shuffle_smiles,palette={1:'#6D042C',0:"#EECF63",2:"#3E7E6C",3:"#C10A36"},s=35)
sns.lineplot(np.arange(-3,2.),np.arange(-3,2.),color='r')
plt.xlabel('Measured $logRCF_{soil}$')
plt.ylabel('Predicted $logRCF_{soil}$')
plt.text(-3.1,1,"R-squared = %0.2f" % np.mean(test_score_all_w_smiles),ha='left',va='top')
plt.legend(loc='lower right')
#uncomment code below to ouput and save figure, there may be slightly difference due to random state and/or package versions
#plt.savefig('gbrt_r2_final',dpi=600)



print('test r2 for GBRT-ECFP model is', np.mean(test_score_all_w_smiles))
print('GBRT-ECFP Finished')



mean_feature_importance_all_impurity_smiles = np.mean(feature_importance_all_w_smiles,0)
sorted_feature_imporatnce_idx_impurity_smiles = np.argsort(mean_feature_importance_all_impurity_smiles[:-2])[::-1]
#there may be slightly difference due to random state and/or package versions
top_10_impurity = sorted_feature_imporatnce_idx_impurity_smiles[:10]
print('The top 10 most important substructure id from impurity importance is',top_10_impurity)


mean_feature_importance_all_permute_smiles_test = np.mean(permute_importance_all_w_smiles_test,0)
sorted_feature_imporatnce_idx_permute_smiles_test = np.argsort(mean_feature_importance_all_permute_smiles_test[:-2])[::-1]
top_10_permute = sorted_feature_imporatnce_idx_permute_smiles_test[:10]
print('The top 10 most important substructure id from permutation importance is',top_10_permute)


#============================run below codes for linear regression model=============================

print('running linear regression model')
total_id = np.arange(n_sample)
np.random.shuffle(total_id)

train_split_index,test_split_index = Kfold(n_sample,5)

splits = 5
prediction_lr = []
prediction_true_lr = []
test_score_all_lr = []
for k in range(splits):
    
    print('split is ',k)
    
    train_id = [total_id[i] for i in train_split_index[k]]
    test_id = [total_id[i] for i in test_split_index[k]]
    
    train_feature = [feature_wo_smiles_z[i] for i in train_id]
    train_label = [RCF_soil[i] for i in train_id]
    
    test_feature = [feature_wo_smiles_z[i] for i in test_id]
    test_label = [RCF_soil[i] for i in test_id]
    
    model = LinearRegression()
    model.fit(np.array(train_feature),np.array(train_label).reshape(-1))
    test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
    pred = model.predict(test_feature)
     
    print('test score for this split',test_score)
    prediction_lr.append(pred)
    prediction_true_lr.append(test_label)
    test_score_all_lr.append(test_score)


print('test r2 for simple linear regression model is', np.mean(test_score_all_lr))


prediction_wo_smiles_lr = []
for l in prediction_lr:
    for v in l:
        prediction_wo_smiles_lr.append(v)



prediction_true_wo_smiles_lr = []
for l in prediction_true_lr:
    for v in l:
        prediction_true_wo_smiles_lr.append(v)



ccmap_shuffle_linear = []
for i in total_id:
    ccmap_shuffle_linear.append(ccmap[i])



#===========following codes plot results for linear regression model===========================


sns.scatterplot(prediction_true_wo_smiles_lr,prediction_wo_smiles_lr,hue=ccmap_shuffle_linear,palette={1:'#6D042C',0:"#EECF63",2:"#3E7E6C",3:"#C10A36"},s=35)
sns.lineplot(np.arange(-3,2.0),np.arange(-3,2.0),color='r')
plt.xlabel('Measured $logRCF_{soil}}$')
plt.ylabel('Predicted $logRCF_{soil}$')
plt.text(-3.1,1,"R-squared = %0.2f" % np.mean(test_score_all_lr),ha='left',va='top')
plt.legend(loc='lower right')

#uncomment below code for outputing and saving the figure
#plt.savefig('linear_r2_final',dpi=600)
print('Finish LR')
