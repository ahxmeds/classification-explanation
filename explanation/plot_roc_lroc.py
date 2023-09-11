#%%
import pandas as pd 
import numpy as np
import os 
import glob 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, roc_auc_score
#%%
def find_tpr_fpr(data, t):
    positive = data[data['truth'] == 1]
    tp = positive[positive['proba'] >= t]
    negative = data[data['truth'] == 0]
    fp = negative[negative['proba'] >= t]
    n_positives = len(positive)
    tpr = len(tp)/len(positive)
    fpr = len(fp)/len(negative)
    imageids_tp = tp['ImageID'].values
    datasetids_tp = tp['DatasetID'].values
    return tpr, fpr, n_positives, imageids_tp, datasetids_tp

def get_required_localization_data(localization_data, imageids):
    required_localization_data = localization_data[localization_data['ImageID'].isin(imageids)] 
    return required_localization_data
# %%
data  = pd.read_csv('truth_probs.csv')
camtype = 'random'
sdir = '/home/shadab/Projects/classification-explanation/explanation/empirical_lroc_lastlayer'
os.makedirs(sdir, exist_ok=True)
localization_fpath = os.path.join(sdir, f"{camtype}_empirical_lroc.csv")
localization_data = pd.read_csv(localization_fpath)
#%%
import time 
start = time.time()
thresholds = np.linspace(1,1e-9,3500)
TPR = []
FPR = []
TPR_correctlocal = []

for t in thresholds:
    tpr, fpr, n_positives, imageids_tp, datasetids_tp = find_tpr_fpr(data, t)
    
    localization_tp = get_required_localization_data(localization_data, imageids_tp)
    n_tp_correctlocal = len(localization_tp[localization_tp['TPR r=20']>=0.5])
    tpr_correctlocal = n_tp_correctlocal/n_positives

    TPR.append(tpr)
    FPR.append(fpr)
    TPR_correctlocal.append(tpr_correctlocal)
    print(f"Done with threshold: {t}")

#%%
fig, ax = plt.subplots()
ax.plot(FPR, TPR)
auc_roc = np.trapz(TPR, FPR)
ax.plot(FPR, TPR_correctlocal)
auc_lroc = np.trapz(TPR_correctlocal, FPR)
plt.show()
print(f"AUC ROC: {auc_roc}")
print(f"AUC LROC: {auc_lroc}")
# %%
savedir = '/home/shadab/Projects/classification-explanation/explanation/lroc_plot_data'
os.makedirs(savedir, exist_ok=True)
savedata = np.column_stack((FPR, TPR_correctlocal))
savedata_df = pd.DataFrame(savedata, columns=['FPR', 'TPR_correctlocal'])
savepath = os.path.join(savedir, f"{camtype}_lroc_plot_data.csv")
savedata_df.to_csv(savepath, index=False)
print(f"{(time.time() - start)/60} mins") 
# %%
# savedata = np.column_stack((FPR, TPR))
# savedata_df = pd.DataFrame(savedata, columns=['FPR', 'TPR'])
# savepath = os.path.join(savedir, f"roc_plot_data.csv")
# savedata_df.to_csv(savepath, index=False)
# %%
