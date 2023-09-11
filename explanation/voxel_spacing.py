#%%
import numpy as np 
import SimpleITK as sitk 
import os 
import glob 
# %%
def read_image_array(path):
    image = sitk.ReadImage(path)
    return image
#%%
dir1 = '/data/blobfuse/lymphoma_lesionsize_split/dlbcl_bccv/all/labels'
dir2 = '/data/blobfuse/lymphoma_lesionsize_split/pmbcl_bccv/all/labels'
dir3 = '/data/blobfuse/lymphoma_lesionsize_split/dlbcl_smhs/all/labels' 
# %%
paths1 = sorted(glob.glob(os.path.join(dir1, '*.nii.gz')))
paths2 = sorted(glob.glob(os.path.join(dir2, '*.nii.gz')))
paths3 = sorted(glob.glob(os.path.join(dir3, '*.nii.gz')))
# %%
paths = paths1 + paths2 + paths3
# %%
voxels = []


for path in paths:
    image = sitk.ReadImage(path)
    voxels.append(image.GetSpacing())


# %%
x_voxelsize = [voxels[i][0] for i in range(len(voxels))]
print(np.median(x_voxelsize))
# %%
