#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
# %%
def plot_LROC(lroc_fpaths, legend_labels):
    dfs = [pd.read_csv(fpath) for fpath in lroc_fpaths]
    fig, ax = plt.subplots()
    AUCs = []
    FPR = dfs[0]['FPR'].values
    TPR = dfs[0]['TPR'].values
    ax.plot(FPR, TPR)
    auc_roc = np.trapz(TPR, FPR)
    AUCs.append(auc_roc)
    for df in dfs[1:]:
        FPR = df['FPR'].values
        TPR_correctlocal = df['TPR_correctlocal'].values
        ax.plot(FPR, TPR_correctlocal)
        auc_lroc = np.trapz(TPR_correctlocal, FPR)
        AUCs.append(auc_lroc)
    legend_labels_auc = [f"{legend_labels[i]}: AUC: {round(AUCs[i],4)}" for i in range(len(legend_labels))]
    ax.legend(legend_labels_auc, bbox_to_anchor=(0.8, -0.2))
    ax.set_xlabel('False positive rate (FPR)')
    ax.set_ylabel('True positive rate (TPR)')
    ax.set_title('LROC at R = 6.2 cm (20 pixels)')
    plt.show()


#%%
def plot_empirical_LROC(lroc_fpaths, legend_labels, vln):
    dfs = [pd.read_csv(path) for path in lroc_fpaths]
    radii = np.arange(1,126)

    rcolnames= [f'TPR r={int(radii[i])}' for i in range(len(radii))]
    proportion_means = []
    proportion_stds = []
    for df in dfs:
        prop_vals =  [df[rcolnames[i]].astype(float).mean() for i in range(len(rcolnames))]
        proportion_means.append(prop_vals)
        prop_stds =  [df[rcolnames[i]].astype(float).std() for i in range(len(rcolnames))]
        proportion_stds.append(prop_stds)
    
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.7)
    
    for i in range(len(dfs)):
        ax.plot(radii, proportion_means[i])
        
    ax.vlines(x=vln, ymin=0.004, ymax=0.8, linestyles='dashed')
    ax.legend(legend_labels)
    ax.set_xlabel('Radius (Pixels)')
    ax.set_ylabel('Fraction localized')
    ax.set_title('Empirical LROC')
    fig.savefig('empirical_lroc.png', dpi=400, bbox_inches='tight')

    
    print(radii[vln-1])
    print(f"{legend_labels[0]}: {round(proportion_means[0][vln-1],3)} +/- {round(proportion_stds[0][vln-1],3)}")
    print(f"{legend_labels[1]}: {round(proportion_means[1][vln-1],3)} +/- {round(proportion_stds[1][vln-1],3)}")
    print(f"{legend_labels[2]}: {round(proportion_means[2][vln-1],3)} +/- {round(proportion_stds[2][vln-1],3)}")
    print(f"{legend_labels[3]}: {round(proportion_means[3][vln-1],3)} +/- {round(proportion_stds[3][vln-1],3)}")
    print(f"{legend_labels[4]}: {round(proportion_means[4][vln-1],3)} +/- {round(proportion_stds[4][vln-1],3)}")
    # print(f"{legend_labels[5]}: {round(proportion_means[5][vln-1],3)} +/- {round(proportion_stds[5][vln-1],3)}")

    print('\n')
    idx = 10
    print(radii[idx-1])
    print(f"{legend_labels[0]}: {round(proportion_means[0][idx-1],3)} +/- {round(proportion_stds[0][idx-1],3)}")
    print(f"{legend_labels[1]}: {round(proportion_means[1][idx-1],3)} +/- {round(proportion_stds[1][idx-1],3)}")
    print(f"{legend_labels[2]}: {round(proportion_means[2][idx-1],3)} +/- {round(proportion_stds[2][idx-1],3)}")
    print(f"{legend_labels[3]}: {round(proportion_means[3][idx-1],3)} +/- {round(proportion_stds[3][idx-1],3)}")
    print(f"{legend_labels[4]}: {round(proportion_means[4][idx-1],3)} +/- {round(proportion_stds[4][idx-1],3)}")
    # print(f"{legend_labels[5]}: {round(proportion_means[5][vln-1],3)} +/- {round(proportion_stds[5][vln-1],3)}")

    print('\n')
    idx = 25
    print(radii[idx-1])
    print(f"{legend_labels[0]}: {round(proportion_means[0][idx-1],3)} +/- {round(proportion_stds[0][idx-1],3)}")
    print(f"{legend_labels[1]}: {round(proportion_means[1][idx-1],3)} +/- {round(proportion_stds[1][idx-1],3)}")
    print(f"{legend_labels[2]}: {round(proportion_means[2][idx-1],3)} +/- {round(proportion_stds[2][idx-1],3)}")
    print(f"{legend_labels[3]}: {round(proportion_means[3][idx-1],3)} +/- {round(proportion_stds[3][idx-1],3)}")
    print(f"{legend_labels[4]}: {round(proportion_means[4][idx-1],3)} +/- {round(proportion_stds[4][idx-1],3)}")
    # print(f"{legend_labels[5]}: {round(proportion_means[5][vln-1],3)} +/- {round(proportion_stds[5][vln-1],3)}")

    return fig, ax

# %%
legend_labels=['GradCAM', 'GradCAM++', 'EigenCAM', 'LayerCAM', 'Random Localizer']
dir = '/home/shadab/Projects/classification-explanation/explanation/empirical_lroc_lastlayer'
gradcampath = os.path.join(dir, 'gradcam_empirical_lroc.csv')
gradcampluspluscampath = os.path.join(dir, 'gradcamplusplus_empirical_lroc.csv')
eigencampath = os.path.join(dir, 'eigencam_empirical_lroc.csv')
layercampath = os.path.join(dir, 'layercam_empirical_lroc.csv')
randompath = os.path.join(dir, 'random_empirical_lroc.csv')
allpaths = [gradcampath,gradcampluspluscampath, eigencampath, layercampath, randompath]
plot_empirical_LROC(allpaths, legend_labels, vln=20)

# %%
legend_labels=['ROC', 'LROC-GradCAM', 'LROC-GradCAM++', 'LROC-EigenCAM', 'LROC-LayerCAM', 'LROC: Random Localizer']#, 'Random Localizer']
dir = '/home/shadab/Projects/classification-explanation/explanation/lroc_plot_data'
rocpath = os.path.join(dir, 'roc_plot_data.csv')
gradcamlrocpath = os.path.join(dir, 'gradcam_lroc_plot_data.csv')
gradcampluspluslrocpath = os.path.join(dir, 'gradcamplusplus_lroc_plot_data.csv')
eigencamlrocpath = os.path.join(dir, 'eigencam_lroc_plot_data.csv')
layercamlrocpath = os.path.join(dir, 'layercam_lroc_plot_data.csv')
randomlrocpath = os.path.join(dir, 'random_lroc_plot_data.csv')
allpaths = [rocpath, gradcamlrocpath, gradcampluspluslrocpath, eigencamlrocpath, layercamlrocpath, randomlrocpath]
plot_LROC(allpaths, legend_labels)
# %%
