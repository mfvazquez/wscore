#!/usr/bin/env python

import os
import numpy as np
import Wmaps


###############################################
## INPUTS
###############################################

# 1. csv_file (REQUIRED)                -- CSV file containing paths to images. First row must contain column names: Filename, Diagnosis
# 2. mask (OPTIONAL but recommended)    -- Path to image mask. If not specified, a mask will be created by binarizing averaged images over 0.1
# 3. model_directory (REQUIRED)         -- Path to W model
# 3. output directory (REQUIRED)        -- W-map images will be placed in "wmaps" folder inside output directory
#
# Note: you may have to set exectuable permissions for wmap_to_stats.sh, wscore_binarized_to_stats.sh, and wscore_mask_to_stats.sh
# located in $MAC/Image_tools
#   e.g chmod +x < script name >

# Example study

# Cross sectional gene carriers
# csv_file = '/mnt/production/projects/MAC-LONI/Mar_2018_rebuild/Make_wmaps_x_sec_all_data_merged_2018-05-10.csv'

# Longitudinal gene carriers
csv_file        = 'segment_csv.csv'
mask            = 'example_mask.nii'
model_directory = 'output'
out_directory   = model_directory


# ARTAG
# csv_file = '/mnt/macdata/groups/imaging_core/matt/ARTAG_project_Elisa/Make_W_maps_WM.csv'
# mask = '/mnt/macdata/groups/imaging_core/matt/ARTAG_project_Elisa/mask_80_split0001.nii.gz'
# out_directory = '/mnt/macdata/groups/imaging_core/matt/ARTAG_project_Elisa/Wmaps_WM_redo'

# Cross sectional
#csv_file      = '/mnt/macdata/groups/imaging_core/matt/projects/mac-loni/June_2018_rebuild/wmaps_cross_sectional_M1/Make_Wmaps_2018-07-17.csv'
#out_directory = '/mnt/macdata/groups/imaging_core/matt/projects/mac-loni/June_2018_rebuild/wmaps_cross_sectional_M1'

# Longitudinal change maps
#csv_file      = '/mnt/production/projects/MAC-LONI/June_2018/Change_maps/Make_W_maps_Change_maps_final_dataset.csv'
#out_directory = '/mnt/production/projects/MAC-LONI/June_2018/Change_maps/w_maps'



###############################################
## WMAPS
###############################################
prod = Wmaps.Preproc()
prod.load_data(csv = csv_file)
#------------------------------------------
#----Step 1. Create Model from controls----
#------------------------------------------

Wmaps.create_model(dataframe_controls = prod.dataframe_controls, model_directory = model_directory, mask = mask)  # if no mask, set mask = None

#------------------------------------------
#----Step 2. Create W-maps from model------
#------------------------------------------

Wmaps.create_WMAP(dataframe_cases = prod.dataframe_cases, model_directory = model_directory, out_directory = out_directory)

#------------------------------------------------------------------
#----Step 3. (OPTIONAL) Create wscore masks and binarized masks----
#------------------------------------------------------------------

    # Provide lists of lower bounds and upper Threshold values
upper_bound = ['2.0']           # ['-0.5', '-0.4'] # upper bound
lower_bound = ['-1.0']          # ['-7.0']         # lower bound

Wmaps.threshold_wmaps(lower_bound = lower_bound, upper_bound = upper_bound, out_directory = out_directory)