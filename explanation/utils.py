#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 08:00:01 2021

@author: shadab
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import nrrd as nrrd
import glob
import os
from skimage.measure import find_contours
import matplotlib.patches as patches
import matplotlib.image as mpimg
import nibabel as nib
from PIL import Image


# convert 1, 2 digit numbers to 3 digits using zeros in front
def convert_to_3_digits(num):
    num = str(num)
    
    if len(num) == 1:
        new_num = '00' + num
    elif len(num) == 2:
        new_num = '0' + num
    else:
        new_num = num   
    return new_num #returns a string



# convert niftii to numpy array and return voxel dims in mm.
def nib2numpy(filepath):
    img = nib.load(filepath)
    voxel_dim = img.header.get_zooms() # in millimeters
    array3d = np.array(img.dataobj)
    return array3d, voxel_dim


# get shape of a NIFTII array (3D or 2D)
def get_shape(path):
    array, voxdim = nib2numpy(path)
    return np.shape(array)

# get patient ID from filepath (won't always work)

def get_patientID(filepath, image_type = 'PET'):
    fileID = filepath.replace(os.path.dirname(filepath)+'/', '')
    
    if image_type == 'PET':
        patientID = fileID[:-8]
        
    elif image_type == 'MASK':
        patientID = fileID[:-9]
    elif image_type == 'CT':
        patientID = fileID[:-7]   
    
    return patientID



#count number of slices in a given 3D nii file

def count_slices(filepath, cross_section = 'coronal'):
    array3d, voxdim = nib2numpy(filepath)
    
    if cross_section == 'sagittal':
        num_slices = array3d.shape[0]
    elif cross_section == 'coronal':
        num_slices = array3d.shape[1]
    elif cross_section == 'axial':
        num_slices = array3d.shape[2]
    
    return num_slices


# get a 3D nii image and save all its 2D slices (as per the cross-section specified) 
#into a folder with the name as patient ID 

def _3d_to_2d(filepath, src_folder, dest_folder, cross_section='coronal', image_type = 'PET'):
    fileID = (filepath.replace(src_folder, ""))
    if image_type == 'PET':
        patientID = fileID[:-8]
    elif image_type == 'MASK':
        patientID = fileID[:-9]
    elif image_type == 'CT':
        patientID = fileID[:-7]
        
    save_folder = dest_folder + str(patientID) + '/'
    os.makedirs(save_folder)
     
    img = nib.load(filepath)
    array3d = np.array(img.dataobj)
    
    if cross_section == 'sagittal':
        for i in range(np.shape(array3d)[0]):
            array2d = array3d[i,:,:]
            img = nib.Nifti1Image(array2d, affine=np.eye(4))
            nib.save(img, save_folder + patientID + '_' + convert_to_3_digits(i) + '.nii')
    
    elif cross_section == 'coronal':
        for i in range(np.shape(array3d)[1]):
            array2d = array3d[:,i,:]
            img = nib.Nifti1Image(array2d, affine=np.eye(4))
            nib.save(img, save_folder + patientID + '_' + convert_to_3_digits(i) + '.nii')
         
    elif cross_section == 'axial':
        for i in range(np.shape(array3d)[2]):
            array2d = array3d[:,:,i]
            img = nib.Nifti1Image(array2d, affine=np.eye(4))
            nib.save(img, save_folder + patientID + '_' + convert_to_3_digits(i) + '.nii')
    


# select slices with lesion for a particular 3D patient MASK (.nii file) and saves the 
# corresponding nii files into the dest folder

def select_slices(filepath, cross_section='coronal'):
    
    lesion_slices = []
    non_lesion_slices = []
    
    array3d, voxdim = nib2numpy(filepath)
    
    
    if cross_section == 'sagittal':
        for i in range(np.shape(array3d)[0]):
            array2d = array3d[i,:,:]
            if np.all(array2d == 0):
                non_lesion_slices.append(i)
            else:
                lesion_slices.append(i)
                
    
    elif cross_section == 'coronal':
        for i in range(np.shape(array3d)[1]):
            array2d = array3d[:,i,:]
            if np.all(array2d == 0):
                non_lesion_slices.append(i)
            else:
                lesion_slices.append(i)
                            
         
    elif cross_section == 'axial':
        for i in range(np.shape(array3d)[2]):
            array2d = array3d[:,:,i]
            if np.all(array2d == 0):
                non_lesion_slices.append(i)
            else:
                lesion_slices.append(i)
    
    return lesion_slices, non_lesion_slices
                    

# rotate the 2D images to correct clinical orientations 
def get_clinical_orientation(array2d, cross_section='axial'):
    
    if cross_section == 'coronal' or cross_section == 'sagittal':
        array2d_rotated = np.rot90(array2d)
    else:
        array2d_rotated = np.fliplr(np.rot90(array2d, k=3))
    
    return array2d_rotated
            
    
# takes a 3D nii image and list of slice IDs and saves the corresponding 2D nii images into 
# dest_folder
def save_selected_slices(array3d, disease, center, patientID, caseID, selected_slices, dest_folder, cross_section='coronal'):

    if cross_section == 'sagittal':
        for i in range(len(selected_slices)):
            array2d = array3d[selected_slices[i],:,:]
            array2d_rotated = get_clinical_orientation(array2d, cross_section='sagittal')
            img = nib.Nifti1Image(array2d_rotated, affine=np.eye(4))
            save_filename = dest_folder.joinpath(disease + '_' + center + '_' + patientID + '_' + caseID + '_' + convert_to_3_digits(selected_slices[i]) + '_sg.nii' )
            nib.save(img, save_filename)
                
    
    elif cross_section == 'coronal':
        for i in range(len(selected_slices)):
            array2d = array3d[:,selected_slices[i],:]
            array2d_rotated = get_clinical_orientation(array2d, cross_section='coronal')
            img = nib.Nifti1Image(array2d_rotated, affine=np.eye(4))
            save_filename = dest_folder.joinpath(disease + '_' + center + '_' + patientID + '_' + caseID + '_' + convert_to_3_digits(selected_slices[i]) + '_co.nii' )
            nib.save(img, save_filename)
         
    elif cross_section == 'axial':
        for i in range(len(selected_slices)):
            array2d = array3d[:,:,selected_slices[i]]
            array2d_rotated = get_clinical_orientation(array2d, cross_section='axial')
            img = nib.Nifti1Image(array2d_rotated, affine=np.eye(4))
            save_filename = dest_folder.joinpath(disease + '_' + center + '_' + patientID + '_' + caseID + '_' + convert_to_3_digits(selected_slices[i]) + '_ax.nii' )
            nib.save(img, save_filename)


# get a 1D array containing all the non-zero elements of a 2D array
def get_non_zero_elements_array(array2d):
    array1d = array2d.flatten()
    array1d_non_zero = np.array([], dtype=np.float32)

    for i in range(len(array1d)):
        if array1d[i] != 0:
            array1d_non_zero = np.append(array1d_non_zero, array1d[i])
    
    return array1d_non_zero


# performs z-score normalization on nii image and saves the new nii image in 
# dest_folder
def z_score_norm(filepath, dest_folder=''):
    fileID = filepath.replace(os.path.dirname(filepath)+'/', '')
    array2d, voxdim = nib2numpy(filepath)
    mean = np.mean(array2d)
    std = np.std(array2d)
    new_array2d = (array2d - mean)/std
    img = nib.Nifti1Image(new_array2d, affine=np.eye(4))
    nib.save(img, dest_folder + fileID) 
    

# performs z-score normalization only for non-zero pixels on nii images and 
# saves the new nii image in dest_folder 
def z_score_norm_nonzero_pixel(filepath, dest_folder=''):
    fileID = filepath.replace(os.path.dirname(filepath)+'/', '')
    array2d, voxdim = nib2numpy(filepath)
    
    # replace zeros with NaN 
    array2d[array2d == 0] = np.nan
    
    # find mean and std ignoring NaN values
    mean = np.nanmean(array2d)
    std = np.nanstd(array2d)
    
    # find norm ignoring NaN values
    array2d = (array2d - mean)/std
    
    array2d = np.nan_to_num(array2d)

    img = nib.Nifti1Image(array2d, affine=np.eye(4))
    nib.save(img, dest_folder + fileID) 
    

# takes a 2D nii file and converts it to jpg
def nii2jpg(filepath, dest_folder=''):
    fileID = filepath.replace(os.path.dirname(filepath)+'/', '')[:-4]
    data, voxdim = nib2numpy(filepath)
    _slice = data
    slice = (_slice.astype(np.float64)-_slice.min()) / (_slice.max()-_slice.min())
    slice = slice.astype('float32')
    cv2.imwrite(dest_folder + fileID + ".jpg", slice*255, [cv2.IMWRITE_JPEG_QUALITY, 100])
    


#takes a YOLO format annotation file (.txt) and count the number of lesions
def count_lesion_in_slice(filename):
    bboxes = np.loadtxt(filename)
    
    if bboxes.ndim == 1:
        return 1
    else:
        return len(bboxes)
    
def get_required_paths_3d(all_paths, required_ids):
    required_paths = []
    for path in all_paths:
        patientID = os.path.basename(path)[:-10]
        if patientID in required_ids:
            required_paths.append(path)

    return required_paths

def get_required_paths_2d(all_paths, required_ids):
    required_paths = []
    for path in all_paths:
        patientID = os.path.basename(path)[:-11]
        if patientID in required_ids:
            required_paths.append(path)

    return required_paths

# rotate 2D images to correct orientation to as displayed     
    
    
        
# if __name__ == '__main__': 
    # images_folder = '/home/shadab/Downloads/niftii_files/MASK_new/MASK_resampled/'
    # images_path = sorted(glob.glob(r"/home/shadab/Downloads/niftii_files/MASK_new/MASK_resampled/*.nii"))
    
    # csvname = 'MASK_info.csv'
    # file = open(csvname, 'w+')
    # file.write("Patient ID,Image size,Voxel dim\n")
    
    # for path in images_path:
    #     fileID = (path.replace(images_folder, ""))
    #     patientID = get_patientID(path, image_type='MASK')
    #     array3d, voxel_dim = nib2numpy(path)
    #     file.write(patientID + ',(' + str(array3d.shape[0]) + " " + str(array3d.shape[1]) + " " + str(array3d.shape[2]) + "),(" + str(voxel_dim[0]) + " " + str(voxel_dim[1]) + " " + str(voxel_dim[2]) + ")\n")
    #     print('Done with patientID: ', patientID)
        
        
    # file.close()
    
    # filepath = '/home/shadab/Downloads/niftii_files/PET/34886217_PET.nii'
    # array, voxdim = nib2numpy(filepath) 
    
    # folder = '/home/shadab/Downloads/scripts/try/'
    

    # for i in range(np.shape(array)[2]):
    #     # array2d = arr[:,i,:]
    #     img = nib.Nifti1Image(array[:,:,i], affine=np.eye(4))
    #     nib.save(img, folder+'img_' + convert_to_3_digits(i) + '.nii')
    #     # plt.imshow(arr[i,:,:])
    #     print(i)
    
    # fig, ax = plt.subplots(1,2, figsize=(30,10))
    # count = 0    
    # images_path = sorted(glob.glob(r"/home/shadab/Downloads/scripts/try/*.nii"))
    # for path in images_path:
    #     arr, vox = nib2numpy(path)
    #     ax[0].imshow(array[:,:,count])
    #     # print("array_printed")
    #     ax[1].imshow(arr)
    #     # print("arr printed")
        
    #     count += 1
       
    # plt.show()
    
    # filepath = sorted(glob.glob('/home/shadab/Downloads/niftii_files/PET/*.nii'))
    # src_folder = '/home/shadab/Downloads/niftii_files/PET/'
    # dest_folder = '/home/shadab/Downloads/niftii_files/PET_2D_coronal_slices/'
    # cross_section = 'coronal'
    
    
    # for path in filepath:
    #     _3d_to_2d(path, src_folder, dest_folder, cross_section, image_type)
    

    # slices_info_dict = {}
    
    # images_path = sorted(glob.glob('//home/shadab/Downloads/niftii_files/MASK_new/MASK_resampled/*.nii'))
    # csvfname = 'patient_slices_count_axial_new.csv'
    # fcsv = open(csvfname, 'w')
    # fcsv.write("Patient ID,slices count, lesion slices count, non lesion slices count\n")
   
    
    # for path in images_path:
        
    #     patientID = get_patientID(path, image_type='M')
        
    #     num_slices = count_slices(path, cross_section='axial')
        
    #     lesion_slices, non_lesion_slices = select_slices(path, cross_section='axial')
        
    #     fcsv.write(patientID + ',' + str(int(num_slices)) + ',' + str(len(lesion_slices)) + ',' + str(len(non_lesion_slices)) + '\n')
    #     slices_info_dict[patientID] = {'Lesion':lesion_slices, 'NonLesion': non_lesion_slices}
    
    #     print(f"Done with: ", patientID)
    
    
    # dictfile = 'slices_info_dict_axial.npy'

    # np.save(dictfile, slices_info_dict)
    
    # fcsv.close()
    
    
    # D = np.load('slices_info_dict_axial.npy',allow_pickle='TRUE').item()
    
    
    
    # images_path = sorted(glob.glob(r"/home/shadab/Downloads/niftii_files/PET_new/PET/*.nii"))
    # lesion_folder = '/home/shadab/Downloads/niftii_files/PET_new/PET_2D_axial_slices/lesion_slices/'
    # non_lesion_folder = '/home/shadab/Downloads/niftii_files/PET_new/PET_2D_axial_slices/non_lesion_slices/'
    
    # slices_dict = np.load('slices_info_dict_axial.npy',allow_pickle='TRUE').item()
    
    # for path in images_path:
    #     patientID = get_patientID(path, image_type = 'PET')
    #     lesion_slices = slices_dict[patientID]['Lesion']
    #     non_lesion_slices = slices_dict[patientID]['NonLesion']
        
    #     save_selected_slices(path, lesion_slices, lesion_folder, cross_section='axial', image_type = 'PET')
    #     save_selected_slices(path, non_lesion_slices, non_lesion_folder, cross_section='axial', image_type = 'PET')
        
    #     print("done with patient ID: " + patientID)
    
    
    
    # images_path = sorted(glob.glob(r"/home/shadab/Downloads/niftii_files/MASK_new/MASK_2D_coronal_slices/lesion_slices_norm/*.nii"))
    # dest_folder = '/home/shadab/Downloads/niftii_files/PET_2D_coronal_slices/jpg_non_lesion_slices_norm/'
    # count = 0
    # for path in images_path:
    #     nii2jpg(path, dest_folder)
    #     count += 1
    #     print(count)
        
    
    # bboxes_path = sorted(glob.glob(r"/home/shadab/Downloads/niftii_files/MASK_2D_coronal_slices/lesion_bbox/*.txt"))
    
    # csvfname = 'lesion_count_per_slice.csv'
    # f = open(csvfname, 'w+')
    # f.write('Patient ID, Slice ID, lesion count\n')
    # total = 0
    # for path in bboxes_path:
    #     fileID = path.replace(os.path.dirname(path)+'/', '').split('_')
    #     patientID = fileID[0]
    #     sliceID = fileID[1][:-4]
    #     count = count_lesion_in_slice(path)
    #     total += count
    #     f.write(patientID +',' + sliceID + ',' +  str(int(count)) + '\n')
    
    # print(total)
    # f.close()        
    
    # images_path = sorted(glob.glob(r"/home/shadab/Downloads/niftii_files/MASK_2D_coronal_slices/non_lesion_slices/*.nii"))
    # save_folder = '/home/shadab/Downloads/niftii_files/MASK_2D_coronal_slices/non_lesion_bbox/'
    # count = 0
    # for path in images_path:
    #     fileID = path.replace(os.path.dirname(path)+'/', '')[:-4] + '.txt'
    #     txtfilepath = save_folder + fileID 
    #     f = open(txtfilepath, 'w')
    #     f.write('1 0.5 0.5 0.9 0.9\n')
    #     f.close()
    #     count += 1
    #     print(count)
    
    # images_path = sorted(glob.glob(r"/home/shadab/Downloads/niftii_files/PET_new/PET_2D_axial_slices/non_lesion_slices/*.nii"))
    # dest_folder = '/home/shadab/Downloads/niftii_files/PET_new/PET_2D_axial_slices/non_lesion_slices_norm/'
    # count = 0
    # for path in images_path:
    #     z_score_norm_nonzero_pixel(path, dest_folder)
    #     count += 1
    #     print(count)
    
    
    # images_path = sorted(glob.glob(r"/home/shadab/Downloads/niftii_files/MASK_new/MASK_2D_axial_slices/lesion_slices/*.nii"))
    # dest_folder = '/home/shadab/Downloads/niftii_files/MASK_new/MASK_2D_axial_slices/jpg_lesion_slices/'
    
    # count = 0
    # for path in images_path:
        
    #     nii2jpg(path, dest_folder)
        
        
    #     count += 1
    #     print(count)
        
    
    # D = np.load('slices_info_dict_axial.npy', allow_pickle='TRUE').item()
    # fig1, ax1 = plt.subplots()
    # images_path = sorted(glob.glob(r"/home/shadab/Downloads/niftii_files/MASK_2D_axial_slices/lesion_slices/963063965_169.nii"))
    
    
    # for path in images_path:
    #     data, vox = nib2numpy(path)
        
       
    #     # data2d = data[:,:,i]
    #     ax1.imshow(data, cmap='gray')
    #     ax1.text(10,10, str(169), color='white')
        

    
    # import matplotlib.image as mpimg
    # fig2, ax2 = plt.subplots()
    # pathh = '/home/shadab/Downloads/niftii_files/MASK_2D_axial_slices/testing_fab/test_images/963063965_169.jpg'   
        
    # img = mpimg.imread(pathh)
    # ax2.imshow(img)
    
    # images_path = sorted(glob.glob(r"/home/shadab/Downloads/niftii_files/MASK_new/MASK_2D_axial_slices/non_lesion_slices/*.nii"))
    # save_folder = '/home/shadab/Downloads/niftii_files/MASK_new/MASK_2D_axial_slices/non_lesion_bbox/'
    
    # for path in images_path:
        
    #     filename = save_folder + path.replace(os.path.dirname(path)+'/', '')[:-4] + '.txt'
    #     file = open(filename, 'w+')
    #     file.write('1 0.5 0.5 0.8 0.8\n')
    #     file.close()
    
    
    # pet_img_path = '/home/shadab/Downloads/niftii_files/PET_new/PET_2D_axial_slices/lesion_slices/963063965_157.nii'
    
    # bbox_path = '/home/shadab/Downloads/YOLOv3_scratch/analysis/localization_error/test_bbox/963063965_157.txt'
    
    # data, vox = nib2numpy(pet_img_path)
    # box = np.loadtxt(bbox_path)
    
