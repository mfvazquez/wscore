import shutil
import sys, os
import glob
import nibabel as ni
import numpy as np
import pandas as pd
import subprocess
from sklearn import linear_model
from progressbar import ProgressBar
import pdb


# Revised Wmaps.py
# Original author: Gabriel Marx
#
# This package is used to create W-Map images. It both estimates models and W-scores. 
#
# The create_model module will estimate a W-maps model. This is a necessary first step, but one model can be used to create many W-map images.
#
# The create_WMAP module will create W-map images based on a previously esitmated W-map model.

class bcolors:
    def __init__(self):
        self.OKGREEN_   = '\033[92m'
        self.WARNING_   = '\033[93m'
        self.FAIL_      = '\033[91m'
        self.ENDC_      = '\033[0m'

    def print_message(self, Color, Mess):
        print (Color + "\t {}".format(Mess), self.ENDC_)

    def FAIL(self, Mess):
        self.print_message(self.FAIL_, Mess)

    def WARNING(self, Mess):
        self.print_message(self.WARNING_, Mess)

    def INFO_OK(self, Mess):
        self.print_message(self.OKGREEN_, Mess)

color = bcolors()

class Preproc():
    def __init__(self):
        self.dataframe_controls = ""
        self.dataframe_cases    = "" 

    def load_data(self, csv):
        data_df = pd.read_csv(csv)
        data_df.dropna(inplace=True)
        color.WARNING("Any NaNs will be dropped")
        param_columns = data_df.columns[2:] # First two columns reserved for image paths and Diagnosis
        self.dataframe_controls = data_df.loc[data_df.Diagnosis == 0]
        self.dataframe_cases    = data_df.loc[data_df.Diagnosis >= 1]

def get_4D_cube(df):
    images = []
    for row in df.iterrows():
        subject = row[1][0]
        print (subject)
        img  = ni.load(subject)
        data = img.get_data()
        images.append(data)

    images = np.asarray(images)

    return images

def create_mask(fourD, threshold = 0.1):
    color.WARNING("Warning! Not using a mask is ill-advised. Creating a mask...")
    meanImg = fourD.mean(axis = 0)
    maskImg = (meanImg > threshold).astype(int)

    return maskImg

def save_nifti(data, template, outdir, filename):
    outImg  = ni.Nifti1Image(data, template.affine, header = template.header)
    outfile = os.path.join(outdir, filename)
    outImg.to_filename(outfile)

def fit_model( controlImgs, parameters, mask = None):
    """
    Fit a WMAP model to a bunch of contol images
    :param controlImgs:
    :param parameters:
    :param mask:
    :return: BetaImg (4D array), r2 (4D image), sigma (3D image)
    """

    if mask is not None:
        if not (mask.size == controlImgs[0].size):
            raise ValueError(color.FAIL('The mask size is not the same as the control images'))
        mask = mask.flatten()

    nz_inds = mask.nonzero()[0]
    nnz     = len(nz_inds)

    # first we need to flatten all images into 1D arrays
    numControlPatients, sx, sy, sz = controlImgs.shape
    imgShape = (sx, sy, sz)

    nPatients, numParameters = parameters.shape

    if nPatients != numControlPatients:
        raise ValueError(color.FAIL('The number of patients in the parameters array does not match the number of control images!'))

    flatImages = []

    for img in controlImgs:
        img = img.flatten()
        flatImages.append(img)

    flat = np.asarray(flatImages)

    voxelData    = flat.T
    numVoxels, _ = voxelData.shape
    betaImgs     = np.zeros([numParameters ,numVoxels])
    rSquareImgs  = np.zeros([numParameters ,numVoxels])
    interceptImg = np.zeros(numVoxels)
    scoreImg     = np.zeros([numVoxels])          # r^2 of whole model fit to voxel
    sigmaImg     = np.zeros([numVoxels])

    pbar = ProgressBar()

    color.INFO_OK("Creating model")

    ## for analysis
    full_predictions = np.zeros([nPatients, numVoxels])
    ##

    for i in pbar(range(nnz)):
        voxelNum  = nz_inds[i]
        thisVoxel = voxelData[voxelNum]
        #pdb.set_trace()
        clf = linear_model.LinearRegression()
        clf.fit(parameters, thisVoxel)

        betaImgs[:, voxelNum]    = clf.coef_
        interceptImg[voxelNum]   = clf.intercept_ 
        scoreImg[voxelNum]       = clf.score(parameters, thisVoxel)
        corrMatrix               = np.corrcoef(thisVoxel, parameters.T)
        rs2Matrix                = np.multiply(corrMatrix, corrMatrix)
        rSquareImgs[:, voxelNum] = rs2Matrix[1:, 0]            # skip diagonal 1 in corr matrix
        predictions              = clf.predict(parameters)
        full_predictions[:,i]    = predictions
        residuals                = thisVoxel - predictions
        stdDev                   = residuals.std()
        sigmaImg[voxelNum]       = stdDev

    scoreImg     = np.reshape(scoreImg, imgShape)
    sigmaImg     = np.reshape(sigmaImg, imgShape)
    interceptImg = np.reshape(interceptImg, imgShape)

    betaI = []
    betaI.append(interceptImg)

    for row in betaImgs:
        img = np.reshape(row, imgShape)
        betaI.append(img)

    rSq =[]

    for row in rSquareImgs:
        img = np.reshape(row, imgShape)
        rSq.append(img)

    return betaI, scoreImg, rSq, sigmaImg

def est_W(patientImg, patientParams, betaImgs, sigmaImg):
    numParams = patientParams.shape[0]
    assert(numParams == betaImgs.shape[3] - 1), 'Number of patient parameters is not equal to the number of beta maps for the linear model'
    interceptImg = betaImgs[:,:,:,0]
    betaImgs     = betaImgs[:,:,:,1:]

    predictedImg = interceptImg.copy()
    for i in range(0, numParams):
        predictedImg =  predictedImg + betaImgs[:,:,:,i] * patientParams[i]

    wMap = (patientImg - predictedImg) / ((sigmaImg) + 1e-8)
    wMap[np.isnan(wMap)] = 0
    wMap[np.isinf(wMap)] = 0

    return wMap

def create_model(dataframe_controls, model_directory = None, mask = None ):
    """ This module estimates the w-map models necessary for w-map production.
    INPUT:
        csv (required)           -- a csv file containing paths to your images in the first column and your model parameters in the subsequent columns.
                                    NOTE: the script assumes that the first row is headers...
        model_directory (required) -- this is the directory for your output and where the create_WMAP module will be pointed to.
                                    if left unspecified, it will use the current working directory.
        mask (OPTIONAL)          -- a mask image to confine the voxels of the analysis.
                                    if left unspecified, it will create a mask by averaging all images and thresholding at 0.1
                                    (ONLY SUITABLE FOR CROSS-SECTIONAL VBM and FA - DTI)
    OUTPUT:
        model_directory will be filled with various images which represent the estimated model. In addition, the csv will be copied into the model_directory.
    """

    if model_directory is None:
        color.FAIL("No output directory specified")
        sys.exit(1)
    else:
        if not os.path.exists(model_directory):
            os.mkdir(model_directory)
    if dataframe_controls.shape[0] == 0:
        color.FAIL("Please specify controls with Diagnosis = 0")
        sys.exit(1)
    if mask:
        if not os.path.exists(mask):
            color.FAIL("Error: Cannot find mask")
            sys.exit(1)
        mask_img_ = ni.load(mask)
        mask_img  = mask_img_.get_data()
        shutil.copy(mask, os.path.join(model_directory, 'mask.nii'))
    model_params = dataframe_controls.copy()
    model_params.drop('Diagnosis', axis=1, inplace=True)
    outpath = os.path.join(model_directory,'model_params.csv')
    model_params.to_csv(outpath,index=False)

    print ("\n")
    print ("############################")
    print ("## Loading controls...")
    print ("############################")

    image_array    = get_4D_cube(dataframe_controls)
    parameterNames = dataframe_controls.columns[2:]
    parameters_    = dataframe_controls[parameterNames].values.astype(float)
    parameters     = (parameters_ - np.mean(parameters_, axis=0)) / (np.std(parameters_, axis = 0) + 1e-8)
    temp           = ni.load(dataframe_controls.iloc[0][0])

    if mask is None:
        mask_img = create_mask(image_array)
        save_nifti(mask_img, temp, model_directory, 'mask.nii')

    betaImgs_, scoreImg, rSq_, sigmaImg = fit_model(image_array, parameters, mask = mask_img)
    betaImgs_ = np.asarray(betaImgs_)
    betaImgs  = np.rollaxis(betaImgs_, 0, 4)

    rSq_ = np.asarray(rSq_)
    rSq  = np.rollaxis(rSq_, 0, 4)  # mricron wants the last axis to be the volume number in the deck

    save_nifti(betaImgs, temp, model_directory, 'betaImgs.nii')
    save_nifti(scoreImg, temp, model_directory, 'score.nii')
    save_nifti(sigmaImg, temp, model_directory, 'sigma.nii')
    save_nifti(rSq, temp, model_directory, 'rSq.nii')

def create_WMAP(dataframe_cases, model_directory, out_directory):
    if not os.path.exists(model_directory):
        color.FAIL("Model directory does not exist")
        sys.exit(1)
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
        os.mkdir(os.path.join(out_directory,'wmaps'))
        outdir = os.path.join(out_directory,'wmaps')
    else:
        outdir = os.path.join(out_directory,'wmaps')

    beta_img_      = ni.load(os.path.join(model_directory, 'betaImgs.nii'))
    sigma_img_     = ni.load(os.path.join(model_directory, 'sigma.nii'))
    model_params_  = pd.read_csv(os.path.join(model_directory, 'model_params.csv'))
    model_params   = model_params_[model_params_.columns[1:]].values.astype(float) # Diagnosis col should have been dropped
    beta_img       = beta_img_.get_data()
    sigma_img      = sigma_img_.get_data()
    parameterNames = dataframe_cases.columns[2:]
    parameters_    = dataframe_cases[parameterNames].values.astype(float)
    parameters     = (parameters_ - np.mean(model_params, axis=0)) / np.std(model_params, axis = 0)
    temp           = ni.load(dataframe_cases.iloc[0][0]) # use first case as template

    color.INFO_OK( "Creating WMAPs" )

    for i, subject in enumerate(dataframe_cases.iterrows()):
        filepath  = subject[1][0]
        filename  = os.path.basename(filepath)
        case_img_ = ni.load(filepath)
        case_img  = case_img_.get_data()
        #params    = subject[1][1:].values.astype(float)
        params    = parameters[i]
        outname   = 'W_map_{}'.format(filename)
        print (filename)
        w_map = est_W(case_img, params, beta_img, sigma_img)
        save_nifti(w_map, template = temp, outdir = outdir, filename = outname)

    # convert images to stats
    color.INFO_OK( "Creating stats CSV")
    cmd = "$MAC/Image_tools/W_map_to_stats.sh {:s}".format(outdir)
    subprocess.call(cmd, shell=True)

def wscore_masks(wmaps_dir, upper_thresh, lower_thresh):
    color.INFO_OK("Creating w-score and binarized masks")

    if not os.path.exists(wmaps_dir):
        color.FAIL("wmaps_dir does not exist")
        sys.exit()
    # directories
    dir_wscore    = os.path.join(wmaps_dir, 'wscore_{:s}_to_{:s}'.format(upper_thresh, lower_thresh))
    dir_mask      = os.path.join(dir_wscore, 'mask')
    dir_binarized = os.path.join(dir_wscore, 'binarized')

    if not os.path.exists(dir_wscore):
        os.mkdir(dir_wscore)
    if not os.path.exists(dir_mask):
        os.mkdir(dir_mask)
    if not os.path.exists(dir_binarized):
        os.mkdir(dir_binarized)

    wmap_list =  os.listdir(wmaps_dir)
    wmap_list = [i for i in wmap_list if '.nii' in i]
    nsubj     = len(wmap_list)

    pbar = ProgressBar()
    for i, p in enumerate(pbar(range(nsubj))):
        wmap = wmap_list[i]
        #pdb.set_trace()
        if 'gz' in wmap:
            base = os.path.splitext(wmap)[0][:-4]
            ext  = '.nii.gz'
        else:
            base = os.path.splitext(wmap)[0]           # 0 base
            ext  = os.path.splitext(wmap)[1] + '.gz'   # 1 ext
        fname_mask          = base + '_mask' + ext
        fname_binarized     = base + '_binarized' + ext
        full_file_wmap      = os.path.join(wmaps_dir, wmap)
        full_file_mask      = os.path.join(dir_mask, fname_mask)
        full_file_binarized = os.path.join(dir_binarized, fname_binarized)

        cmd1 = 'fslmaths '+ full_file_wmap + ' -uthr ' + upper_thresh + ' -max '+ lower_thresh + ' -mul -1. ' + full_file_mask
        cmd2 = 'fslmaths '+ full_file_wmap + ' -uthr ' + upper_thresh + ' -max '+ lower_thresh + ' -mul -1. -bin ' + full_file_binarized

        os.system(cmd1)
        os.system(cmd2)

    color.INFO_OK( "Creating MASK stats CSV")
    cmd = "$MAC/Image_tools/W_map_to_stats.sh {:s}".format(dir_mask)
    subprocess.call(cmd, shell = True)

    color.INFO_OK( "Creating BINARIZED stats CSV")
    cmd = "$MAC/Image_tools/W_map_to_stats.sh {:s}".format(dir_binarized)
    subprocess.call(cmd, shell=True)

def threshold_wmaps(lower_bound, upper_bound, out_directory):
    wmaps_dir = os.path.join(out_directory, 'wmaps')    # default
    #  Creates masks and binarized masks for each threshold combination from above
    for i in upper_bound:
        for j in lower_bound:
            wscore_masks(wmaps_dir = wmaps_dir, upper_thresh =  i, lower_thresh = j)