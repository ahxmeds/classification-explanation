#%%
import numpy as np 
import pandas as pd 
import SimpleITK as sitk 
import os 
import glob 
from skimage.transform import resize
from skimage.measure import regionprops
import matplotlib.pyplot as plt 
from scipy.ndimage import center_of_mass

#%%    
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

def resize2d_gtarray(inp, dim_x=224, dim_y=224):
    # order = 0 for Nearest neighbour interpolation
    inp_resized = resize(
        inp,
        output_shape=(dim_x, dim_y), 
        order = 0, 
        mode = 'constant', 
        cval=0, 
        anti_aliasing=False,
        preserve_range=True,
    )
    return inp_resized

def resize2d_ptarray(inp, dim_x=224, dim_y=224):
    # order = 1 (default) for Bilinear interpolation
    inp_resized = resize(
        inp,
        output_shape=(dim_x, dim_y), 
        order = 1, 
        mode = 'constant', 
        cval=0, 
    )
    return inp_resized

# %%
def read_image_array(path):
    image = sitk.ReadImage(path)
    array = np.transpose(sitk.GetArrayFromImage(image), (1,0))
    return array

def given_centroid_and_radius_make_circle_mask(centroid, radius):
    matrix_shape = (224, 224)
    matrix = np.zeros(matrix_shape)
    X, Y = np.meshgrid(np.arange(matrix_shape[1]), np.arange(matrix_shape[0]))
    center_x = centroid[1]
    center_y = centroid[0]

    # Compute the Euclidean distance from each point in the matrix to the center
    distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    # Create a circular binary mask
    mask = distances <= radius
    # Set the value of the circle to 1 in the matrix
    matrix[mask] = 1
    return matrix

def get_proportion_localized(gtarray, matrixarray):
    intersection = np.sum(matrixarray[gtarray==1])
    return intersection/(np.sum(gtarray))

def get_tpr_fpr(gtarray, matrixarray):
    total_matrix_array = gtarray.shape[0]*gtarray.shape[1]
    tp = np.sum(matrixarray[gtarray==1])
    positive = np.sum(gtarray)
    pred_positive = np.sum(matrixarray)
    negative = total_matrix_array - positive
    tpr = tp/positive
    fp = pred_positive - tp
    fpr = fp/negative
    return tpr, fpr

def get_thresholded_cam(camarray1, threshold):
    camarray = camarray1.copy()
    # max_intensity = camarray.max()
    camarray[camarray <= threshold] = 0
    camarray[camarray > threshold] = 1
    return camarray

def get_centroid(heatmap):
    weighted_centroid = center_of_mass(heatmap)
    return weighted_centroid

def get_all_positive_imageids(df):
    df_fg = df[df['truth'] == 1]
    fg_imageids = df_fg['ImageID'].values
    fg_datasetids = df_fg['DatasetID'].values
    return fg_imageids, fg_datasetids
#%%
df = pd.read_csv('truth_probs.csv')

imageids_to_run, datasetids_to_run = get_all_positive_imageids(df)
optimal_model_dir = '/data/blobfuse/saved_models/classification/saved_models_resnet18_focalloss_ptlevelsplit_nocenter_nofreeze'
optimal_model_name = 'classification_ep=005.pth'
dbc_fg_ptdir = '/data/blobfuse/DLBCL_bccancer/patient_level_train_test_split/test/axial_data/pt_fg'
dbc_fg_gtdir = '/data/blobfuse/DLBCL_bccancer/patient_level_train_test_split/test/axial_data/gt_fg'

pbc_fg_ptdir = '/data/blobfuse/PMBCL_bccancer/patient_level_train_test_split/test/axial_data/pt_fg'
pbc_fg_gtdir = '/data/blobfuse/PMBCL_bccancer/patient_level_train_test_split/test/axial_data/gt_fg'

dsk_fg_ptdir = '/data/blobfuse/DLBCL_southkorea/patient_level_train_test_split/test/axial_data/pt_fg'
dsk_fg_gtdir = '/data/blobfuse/DLBCL_southkorea/patient_level_train_test_split/test/axial_data/gt_fg'

#%%
camtype = 'scorecam'
dir = f'/data/blobfuse/eanm_lymphoma_data/resnet18_cam_lastlayer/{camtype}'
cam_paths = sorted(glob.glob(os.path.join(dir, '*.npy')))

# %%
ptpaths = []
gtpaths = []
for i in range(len(imageids_to_run)):
    datasetid = datasetids_to_run[i]
    imageid = imageids_to_run[i]
    if datasetid == 'DLBCL_bccancer':
        ptpath = os.path.join(dbc_fg_ptdir, imageid+'.nii.gz')
        gtpath = os.path.join(dbc_fg_gtdir, imageid+'.nii.gz')
    if datasetid == 'PMBCL_bccancer':
        ptpath = os.path.join(pbc_fg_ptdir, imageid+'.nii.gz')
        gtpath = os.path.join(pbc_fg_gtdir, imageid+'.nii.gz')
    if datasetid == 'DLBCL_southkorea':
        ptpath = os.path.join(dsk_fg_ptdir, imageid+'.nii.gz')
        gtpath = os.path.join(dsk_fg_gtdir, imageid+'.nii.gz')
    
    ptpaths.append(ptpath)
    gtpaths.append(gtpath)

# #%%
ptpaths_sorted = []
gtpaths_sorted = []

for i in range(len(cam_paths)):
    cam_path = cam_paths[i]
    cam_imageid  = os.path.basename(cam_path)[:-4]
    # cam_imageid = '_'.join(cam_basename[:-1])

    for j in range(len(gtpaths)):
        ptpath = ptpaths[j]
        gtpath = gtpaths[j]
        imageid = os.path.basename(ptpath)[:-7]
        if imageid == cam_imageid:
            ptpaths_sorted.append(ptpath)
            gtpaths_sorted.append(gtpath)
            break



#%%
# threshold = 0.5
ImageIDs = []
ExpCAMs = []
centroids_X = []
centroids_Y = []
true_positive_rate_overall = []
# false_positive_rate_overall = []

radii = np.arange(1,126)

for i in range(len(gtpaths_sorted)):
    try:
        ptpath = ptpaths_sorted[i]
        gtpath = gtpaths_sorted[i]
        campath = cam_paths[i]
        imageid = os.path.basename(gtpath)[:-7]
        # print(f"Doing imageid: {imageid}")
        
        ptarray = read_image_array(ptpath)
        gtarray = read_image_array(gtpath)
        ptarray_resized = resize2d_ptarray(ptarray)
        gtarray_resized = resize2d_gtarray(gtarray)

        camarray = np.load(campath)
        centroid = get_centroid(camarray)
        centroids_X.append(int(round(centroid[1])))
        centroids_Y.append(int(round(centroid[0])))
        # matrices = []
        true_positive_rate = np.array([])
        # false_positive_rate = np.array([])
        for r in radii:
            matrix2dmask = given_centroid_and_radius_make_circle_mask(centroid, r)
            tpr, _ = get_tpr_fpr(gtarray_resized, matrix2dmask)
            true_positive_rate = np.append(true_positive_rate, tpr)

        true_positive_rate_overall.append(true_positive_rate)
        # false_positive_rate_overall.append(false_positive_rate)

        ImageIDs.append(imageid)
        ExpCAMs.append(camtype)
        # Thresholds.append(round(threshold,4))
        print(f'Done with imageID: {imageid}')
    except:
        pass

    
# %%
data = np.column_stack((ImageIDs, ExpCAMs, centroids_X, centroids_Y, true_positive_rate_overall))
column_names = ['ImageID', 'ExpCAM', 'CenterX', 'CenterY']
for i in range(len(radii)):
    colname = f'TPR r={int(radii[i])}'
    column_names.append(colname)


data_df = pd.DataFrame(data, columns=column_names)
sdir = '/home/jhubadmin/Projects/classification-explanation/explanation/empirical_lroc_lastlayer'
os.makedirs(sdir, exist_ok=True)
spath = os.path.join(sdir, f'{camtype}_empirical_lroc.csv')
data_df.to_csv(spath, index=False)