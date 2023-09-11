#%%
#%%
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchsummary import summary
import numpy as np 
import pandas as pd
import os 
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt 
import nibabel as nib 
import torch
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam import LayerCAM, GradCAM, EigenCAM, ScoreCAM
from matplotlib.patches import Circle
import sys 
from glob import glob
sys.path.append('/home/jhubadmin/Projects/classification-explanation')
import explanation.utils as utils 
from explanation.transformations import resize_2dtensor_nearest, resize_2dtensor_bilinear, stack_slices, pt_preprocess_1
# %%
def save_slice_nifti(array2d, savepath):
    img = nib.Nifti1Image(array2d, affine=np.eye(4))
    nib.save(img, savepath)

def get_all_positive_imageids(df):
    df_fg = df[df['truth'] == 1]
    fg_imageids = df_fg['ImageID'].values
    fg_datasetids = df_fg['DatasetID'].values
    return fg_imageids, fg_datasetids

def get_all_true_positive_imageids(df):
    df_fg = df[df['truth'] == 1]
    df_fg_tp = df_fg[df_fg['proba'] >= 0.5]
    tp_imageids = df_fg_tp['ImageID'].values
    tp_datasetids = df_fg_tp['DatasetID'].values
    return tp_imageids, tp_datasetids

def get_all_false_negative_imageids(df):
    df_fg = df[df['truth'] == 1]
    df_fg_fn = df_fg[df_fg['proba'] < 0.5]
    fn_imageids = df_fg_fn['ImageID'].values
    fn_datasetids = df_fg_fn['DatasetID'].values
    return fn_imageids, fn_datasetids

def convert_to_string(inp):
    return str(inp).replace('.', '-')

def input_pt_preprocess(inp):
    inp = resize_2dtensor_bilinear(inp, 224,224)
    inp = pt_preprocess_1(inp)
    inp = stack_slices(inp)
    inp = np.moveaxis(inp, source=-1, destination=0)
    inp = torch.from_numpy(inp).type(torch.float32)
    inp = torch.unsqueeze(inp, axis=0)
    return inp

def input_gt_preprocess(inp):
    inp = resize_2dtensor_nearest(inp, 224, 224)
    return inp

def cam_threshold(cam_1c, thrs=0.5):
    cam_1c[cam_1c < thrs] = 0
    return cam_1c

def convert_cam_to_mask(cam):
    cam[cam !=0] = 1
    return cam

def dice_score(seg1, seg2):
    dsc = 2.0*np.sum(seg1[seg2==1])/ (np.sum(seg1) + np.sum(seg2))
    return dsc

def intersection_normed_with_segmentation_mask(seg1, seg2):
    # seg 1 = ground truth
    # seg 2 = thresholded cam
    eps = 1e-6
    norm_intersection = np.sum(seg1[seg2==1])/(np.sum(seg1) + eps)
    return norm_intersection

def get_all_true_negative_imageids(df):
    df_bg = df[df['truth'] == 0]
    df_bg_tn = df_bg[df_bg['proba'] < 0.5]
    tn_imageids = df_bg_tn['ImageID'].values
    tn_datasetids = df_bg_tn['DatasetID'].values
    return tn_imageids, tn_datasetids
# %%
df = pd.read_csv('/home/jhubadmin/Projects/classification-explanation/explanation/truth_probs.csv')
imageids_to_run, datasetids_to_run = get_all_positive_imageids(df)
imageids_to_run = list(imageids_to_run)
datasetids_to_run = list(datasetids_to_run)
#%%
optimal_model_dir = '/data/blobfuse/default/saved_models/classification/saved_models_resnet18_focalloss_ptlevelsplit_nocenter_nofreeze'
optimal_model_name = 'classification_ep=005.pth'
dbc_fg_ptdir = '/data/blobfuse/default/DLBCL_bccancer/patient_level_train_test_split/test/axial_data/pt_fg'
dbc_fg_gtdir = '/data/blobfuse/default/DLBCL_bccancer/patient_level_train_test_split/test/axial_data/gt_fg'

pbc_fg_ptdir = '/data/blobfuse/default/PMBCL_bccancer/patient_level_train_test_split/test/axial_data/pt_fg'
pbc_fg_gtdir = '/data/blobfuse/default/PMBCL_bccancer/patient_level_train_test_split/test/axial_data/gt_fg'

dsk_fg_ptdir = '/data/blobfuse/default/DLBCL_southkorea/patient_level_train_test_split/test/axial_data/pt_fg'
dsk_fg_gtdir = '/data/blobfuse/default/DLBCL_southkorea/patient_level_train_test_split/test/axial_data/gt_fg'

dbc_bg_ptdir = '/data/blobfuse/default/DLBCL_bccancer/patient_level_train_test_split/test/axial_data/pt_bg'
dbc_bg_gtdir = '/data/blobfuse/default/DLBCL_bccancer/patient_level_train_test_split/test/axial_data/gt_bg'

pbc_bg_ptdir = '/data/blobfuse/default/PMBCL_bccancer/patient_level_train_test_split/test/axial_data/pt_bg'
pbc_bg_gtdir = '/data/blobfuse/default/PMBCL_bccancer/patient_level_train_test_split/test/axial_data/gt_bg'

dsk_bg_ptdir = '/data/blobfuse/default/DLBCL_southkorea/patient_level_train_test_split/test/axial_data/pt_bg'
dsk_bg_gtdir = '/data/blobfuse/default/DLBCL_southkorea/patient_level_train_test_split/test/axial_data/gt_bg'


cams_dir = '/data/blobfuse/default/eanm_lymphoma_data/resnet18_cam_lastlayer'
gradcam_dir = os.path.join(cams_dir, 'gradcam')
gradcamplusplus_dir =  os.path.join(cams_dir, 'gradcamplusplus')
eigencam_dir =  os.path.join(cams_dir, 'eigencam')
scorecam_dir =  os.path.join(cams_dir, 'scorecam')

gradcam_paths = [os.path.join(gradcam_dir, f"{imageids_to_run[i]}.npy") for i in range(len(imageids_to_run))]
gradcamplusplus_paths = [os.path.join(gradcamplusplus_dir, f"{imageids_to_run[i]}.npy") for i in range(len(imageids_to_run))]
eigencam_paths = [os.path.join(eigencam_dir, f"{imageids_to_run[i]}.npy") for i in range(len(imageids_to_run))]
scorecam_paths = [os.path.join(scorecam_dir, f"{imageids_to_run[i]}.npy") for i in range(len(imageids_to_run))]
def get_centroid(heatmap):
    weighted_centroid = center_of_mass(heatmap)
    return weighted_centroid

#%%
imageids_required = [
    '800156517_20161011_132_ax_fg',
    '936634454_20190621_163_ax_fg',
    '420430641_20200409_188_ax_fg',
    '351588753_20180806_070_ax_fg',
    '17-12133_20170619_199_ax_fg',
    '10-13830_20100629_110_ax_fg'
]

datasetids_required = []
for id in imageids_required:
    idx = imageids_to_run.index(id)
    datasetids_required.append(datasetids_to_run[idx])

#%%
x = 500
for i in range(len(imageids_required)):
    datasetid = datasetids_required[i]
    imageid = imageids_required[i]
    if imageid.endswith('fg'):
        if datasetid == 'DLBCL_bccancer':
            ptpath = os.path.join(dbc_fg_ptdir, imageid+'.nii.gz')
            gtpath = os.path.join(dbc_fg_gtdir, imageid+'.nii.gz')
        if datasetid == 'PMBCL_bccancer':
            ptpath = os.path.join(pbc_fg_ptdir, imageid+'.nii.gz')
            gtpath = os.path.join(pbc_fg_gtdir, imageid+'.nii.gz')
        if datasetid == 'DLBCL_southkorea':
            ptpath = os.path.join(dsk_fg_ptdir, imageid+'.nii.gz')
            gtpath = os.path.join(dsk_fg_gtdir, imageid+'.nii.gz')
    else:
        if datasetid == 'DLBCL_bccancer':
            ptpath = os.path.join(dbc_bg_ptdir, imageid+'.nii.gz')
            gtpath = os.path.join(dbc_bg_gtdir, imageid+'.nii.gz')
        if datasetid == 'PMBCL_bccancer':
            ptpath = os.path.join(pbc_bg_ptdir, imageid+'.nii.gz')
            gtpath = os.path.join(pbc_bg_gtdir, imageid+'.nii.gz')
        if datasetid == 'DLBCL_southkorea':
            ptpath = os.path.join(dsk_bg_ptdir, imageid+'.nii.gz')
            gtpath = os.path.join(dsk_bg_gtdir, imageid+'.nii.gz')


    imageid = os.path.basename(ptpath)[:-7]
    datapt, voxpt = utils.nib2numpy(ptpath)
    datagt, voxgt = utils.nib2numpy(gtpath)

    datapt_resized = resize_2dtensor_bilinear(datapt, 224, 224)
    datagt_resized = resize_2dtensor_nearest(datagt, 224, 224)
    gradcampath = os.path.join(gradcam_dir, f'{imageid}.npy')
    gradcam = np.load(gradcampath)
    center = get_centroid(gradcam)
    circle = Circle((center[1], center[0]), 20, fill=False, edgecolor='black', linewidth=2)

    print(imageid)
    fig, ax = plt.subplots(1, 3, figsize=(12,3), gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)
    pt = ax[0].imshow(datapt_resized, cmap='nipy_spectral')
    gt1 = ax[1].imshow(datagt_resized, cmap='Greys')
    gt2 = ax[2].imshow(datagt_resized)
    cam = ax[2].imshow(gradcam, cmap='nipy_spectral', alpha=0.5)
    ax[0].set_title('PET', fontsize=20)
    ax[1].set_title('GT', fontsize=20)
    ax[2].set_title('GradCAM + COM', fontsize=20)
    fig.colorbar(pt, ax=ax[0])
    fig.colorbar(gt1, ax=ax[1])
    fig.colorbar(cam, ax=ax[2])
    for j in range(3):
        ax[j].set_xticks([])
        ax[j].set_yticks([])
    ax[2].scatter(center[1], center[0], color='red', edgecolor='black', s=80)
    ax[2].add_patch(circle)
    plt.show()
    plt.close('all')

    
    
# %%
