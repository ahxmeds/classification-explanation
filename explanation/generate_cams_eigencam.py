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
import utils
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam import LayerCAM, GradCAM, EigenCAM, ScoreCAM
from transformations import resize_2dtensor_nearest, resize_2dtensor_bilinear, stack_slices, pt_preprocess_1
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
df = pd.read_csv('truth_probs.csv')
imageids_to_run, datasetids_to_run = get_all_positive_imageids(df)
camtype = 'eigencam'
#%%
optimal_model_dir = '/data/blobfuse/saved_models/classification/saved_models_resnet18_focalloss_ptlevelsplit_nocenter_nofreeze'
optimal_model_name = 'classification_ep=005.pth'
dbc_fg_ptdir = '/data/blobfuse/DLBCL_bccancer/patient_level_train_test_split/test/axial_data/pt_fg'
dbc_fg_gtdir = '/data/blobfuse/DLBCL_bccancer/patient_level_train_test_split/test/axial_data/gt_fg'

pbc_fg_ptdir = '/data/blobfuse/PMBCL_bccancer/patient_level_train_test_split/test/axial_data/pt_fg'
pbc_fg_gtdir = '/data/blobfuse/PMBCL_bccancer/patient_level_train_test_split/test/axial_data/gt_fg'

dsk_fg_ptdir = '/data/blobfuse/DLBCL_southkorea/patient_level_train_test_split/test/axial_data/pt_fg'
dsk_fg_gtdir = '/data/blobfuse/DLBCL_southkorea/patient_level_train_test_split/test/axial_data/gt_fg'

dbc_bg_ptdir = '/data/blobfuse/DLBCL_bccancer/patient_level_train_test_split/test/axial_data/pt_bg'
dbc_bg_gtdir = '/data/blobfuse/DLBCL_bccancer/patient_level_train_test_split/test/axial_data/gt_bg'

pbc_bg_ptdir = '/data/blobfuse/PMBCL_bccancer/patient_level_train_test_split/test/axial_data/pt_bg'
pbc_bg_gtdir = '/data/blobfuse/PMBCL_bccancer/patient_level_train_test_split/test/axial_data/gt_bg'

dsk_bg_ptdir = '/data/blobfuse/DLBCL_southkorea/patient_level_train_test_split/test/axial_data/pt_bg'
dsk_bg_gtdir = '/data/blobfuse/DLBCL_southkorea/patient_level_train_test_split/test/axial_data/gt_bg'

# %%
def get_model():
    # load model
    model = models.resnet18(pretrained=True) 
    for param in model.parameters():
        param.requires_grad = True
    # changing avgpool and fc layers
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128,1),
        nn.Sigmoid()
        )
    return model 
model = get_model().to('cuda:0') 
# %%
saved_model_path = os.path.join(optimal_model_dir, optimal_model_name)
model.load_state_dict(torch.load(saved_model_path))
model.eval()
target_layers = [model.layer4[-1]]
target = 1
def get_centroid(heatmap):
    weighted_centroid = center_of_mass(heatmap)
    return weighted_centroid
#%%
savedir = f'/data/blobfuse/eanm_lymphoma_data/resnet18_cam_lastlayer/{camtype}'
os.makedirs(savedir, exist_ok=True)
count = 0
for i in range(len(imageids_to_run[15:17])):
    datasetid = datasetids_to_run[i]
    imageid = imageids_to_run[i]
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
    
    inputpt = input_pt_preprocess(datapt)
    model.eval() 
    output = model(inputpt.to('cuda:0'))
    decision = 'FG' if output.item() >= 0.5 else 'BG'

    cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)

    grayscale_cam = cam(input_tensor=inputpt, targets=[BinaryClassifierOutputTarget(target)], eigen_smooth=True, aug_smooth=True)
    grayscale_cam_1c = grayscale_cam[0, :]
    center = get_centroid(grayscale_cam_1c)

    fig, ax = plt.subplots(1,4)
    ax[0].imshow(datapt_resized, cmap='Greys')
    ax[1].imshow(datagt_resized, cmap='Greys')
    ax[2].imshow(datagt_resized, cmap='Greys')
    ax[2].scatter(center[1], center[0], color='red')
    ax[3].imshow(grayscale_cam_1c, cmap='Greys')
    ax[3].scatter(center[1], center[0], color='red')
    plt.show()
    plt.close('all')

    imagename = f"{imageid}.npy"
    savepath = os.path.join(savedir, imagename)
    np.save(savepath, grayscale_cam_1c)
    print(f"{count}: Done with image: {imageid}")
    count+= 1

# %%
