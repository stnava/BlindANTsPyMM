
__all__ = ['version']

from pathlib import Path
from pathlib import PurePath
import os
import pandas as pd
import math
import os.path
from os import path
import pickle
import sys
import numpy as np
import random
import functools
from operator import mul
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr
import re
import datetime as dt
from collections import Counter
import tempfile
import uuid
import warnings

from dipy.core.histeq import histeq
import dipy.reconst.dti as dti
from dipy.core.gradients import (gradient_table, gradient_table_from_gradient_strength_bvecs)
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import fractional_anisotropy, color_fa
import nibabel as nib

import ants
import antspynet
import antspyt1w
import siq
import tensorflow as tf
import antspymm
import curvanato
import re

from multiprocessing import Pool
import glob as glob

DATA_PATH = os.path.expanduser('~/.blindantspymm/')

def version( ):
    """
    report versions of this package and primary dependencies

    Arguments
    ---------
    None

    Returns
    -------
    a dictionary with package name and versions

    Example
    -------
    >>> import blindantspymm
    >>> blindantspymm.version()
    """
    import pkg_resources
    return {
              'tensorflow': pkg_resources.get_distribution("tensorflow").version,
              'antspyx': pkg_resources.get_distribution("antspyx").version,
              'antspynet': pkg_resources.get_distribution("antspynet").version,
              'antspyt1w': pkg_resources.get_distribution("antspyt1w").version,
              'antspymm': pkg_resources.get_distribution("antspymm").version,
              'blindantspymm': pkg_resources.get_distribution("blindantspymm").version
              }


def widen_summary_dataframe(mydf, description='Label', value='VolumeInMillimeters', skip_first=False):
    """
    Convert a long-format dataframe to a single-row wide-format dataframe.

    Parameters:
    - mydf: pd.DataFrame, long-format dataframe with at least two columns: 
      one categorical identifier (e.g., 'Label') and one value column.
    - description: str, column name that represents the categories (e.g., 'Label').
    - value: str, column name with numerical values to be spread into wide format.
    - skip_first: boolean, skips the first row

    Returns:
    - pd.DataFrame, reshaped into wide format where each category has its own column.
    """

    def convert_float_to_int(df, column):
        if df[column].dtype == float:
            df[column] = df[column].astype(int)
        return df

    if description not in mydf.columns or value not in mydf.columns:
        raise ValueError(f"Dataframe must contain '{description}' and '{value}' columns.")

    mydf=convert_float_to_int(mydf, description )

    # Pivot table to convert long format to wide format, ensuring a single-row result
    df_wide = mydf.pivot_table(index=None, columns=description, values=value, aggfunc='first')

    # Rename columns to include the description prefix
    df_wide.columns = [f"{description}{col}.{value}" for col in df_wide.columns]

    # Reset index to ensure it's a proper DataFrame
    df_wide = df_wide.reset_index(drop=True)

    if skip_first:
        df_wide = df_wide.drop(df_wide.index[0])

    return df_wide


def structural(simg, simg_mask, template, template_mask, template_labels, type_of_transform='antsRegistrationSyNQuick[s]'):
    """
    Perform image registration of a structural image to a template and propagate masks and labels.
    
    Parameters:
    simg : ants.ANTsImage
        The structural image to be registered.
    simg_mask : ants.ANTsImage
        The brain mask for the structural image.
    template : ants.ANTsImage
        The template image to register to.
    template_mask : ants.ANTsImage
        The brain mask for the template.
    template_labels : ants.ANTsImage
        Labeled regions in the template space to be transformed.
    type_of_transform : str, optional
        The registration type (default is 'antsRegistrationSyNQuick[s]').
    
    Returns:
    dict
        A dictionary with registered images and transforms:
        - 'warped_image': Registered structural image
        - 'warped_mask': Registered structural mask
        - 'inverse_warped_labels': Template labels warped to subject space
        - 'forward_transforms': List of forward transformations
        - 'inverse_transforms': List of inverse transformations
        - 'jacobian': Jacobian determinant of the deformation field
        - 'label_geometry_measures': Geometry measures of the warped labels
    """
    # Perform registration
    tmask = ants.iMath( template_mask, 'MD', 10 )
    tbrain = template * template_mask
    tbrain = ants.crop_image(tbrain, tmask )
    tbrain[ tbrain < 0. ] = 0.
    sbrain = simg * simg_mask
    sbrain[ sbrain < 0.0 ] = 0.0
    reg = ants.registration(
        fixed=tbrain,
        moving=sbrain,
        type_of_transform=type_of_transform
    )
    
    inverse_warped_mask = ants.apply_transforms(
        fixed=simg,
        moving=template_mask,
        transformlist=reg['invtransforms'],
        interpolator='nearestNeighbor'
    )

    # for QC purposes
    mydice = ants.label_overlap_measures( inverse_warped_mask, simg_mask )

    # Warp the template labels to subject space using the inverse transform
    inverse_warped_labels = ants.apply_transforms(
        fixed=simg,
        moving=template_labels,
        transformlist=reg['invtransforms'],
        interpolator='nearestNeighbor'
    )
    
    # Compute the Jacobian determinant of the transformation
    jacobian = ants.create_jacobian_determinant_image(
        domain_image=template,
        tx=reg['fwdtransforms'][0],
        do_log=True
    )
    
    # Compute label geometry measures
    label_geometry_measures = ants.label_geometry_measures(inverse_warped_labels)
    label_geometry_measures_w = widen_summary_dataframe( label_geometry_measures, value='VolumeInMillimeters')
    label_geometry_measures_w = pd.concat( 
        [   label_geometry_measures_w,
            widen_summary_dataframe( label_geometry_measures, 
                value='SurfaceAreaInMillimetersSquared') ], axis=1 )
    return {
        'registration': reg,
        'warped_image': reg['warpedmovout'],
        'inverse_warped_mask': inverse_warped_mask,
        'inverse_warped_labels': inverse_warped_labels,
        'forward_transforms': reg['fwdtransforms'],
        'inverse_transforms': reg['invtransforms'],
        'jacobian': jacobian,
        'label_geometry_measures_wide': label_geometry_measures_w,
        'brain_mask_overlap':mydice
    }

def perfusion():
    return None
    
def pet():
    return None


def template_based_labeling(template, template_mask, prior_template, prior_templatelab0, spc=0.66, type_of_transform='SyN'):
    """
    Perform template-based labeling using affine initialization and SyN registration.

    Parameters:
    template : ANTs image
        Template image.
    template_mask : ANTs image
        Template mask image.
    prior_template : ANTs image
        Reference template for registration.
    prior_templatelab0 : ANTs image
        Atlas segmentation image.
    spc : float, optional
        Resampling spacing, default is 0.66.
    type_of_transform : string
        determines the registration class

    Returns:
    dict
        Dictionary containing transformed template and labels.
    """
    timgb = template * template_mask
    timgbc = ants.crop_image(timgb, ants.iMath(template_mask, 'MD', 10))
    timgbc = ants.resample_image(timgbc, resample_params=[spc] * timgbc.dimension)    
    zz = ants.affine_initializer(timgbc, prior_template, search_factor=200, radian_fraction=2, use_principal_axis=True, local_search_iterations=20)
    prior_templatereg = ants.registration(timgbc, prior_template, type_of_transform, initial_transform=zz)
    prior_templatelab = ants.apply_transforms(timgbc, prior_templatelab0, prior_templatereg['fwdtransforms'], interpolator='nearestNeighbor')
    return {'transformed_template': timgbc, 'transformed_labels': prior_templatelab}


def dwi(dimg, simg, simg_mask, simg_labels, dwibval, dwibvec):
    """
    Perform diffusion-weighted imaging (DWI) registration and reconstruction.

    Parameters:
    dimg : ANTs image
        DWI image.
    simg : ANTs image
        Structural image.
    simg_mask : ANTs image
        Structural mask.
    simg_labels : ANTs image
        Structural labels within which we summarize features.
    dwibval : str
        B-value file.
    dwibvec : str
        B-vector file.

    Returns:
    dict
        Dictionary containing registered DWI image and diffusion tensor imaging (DTI) results.
    """
    dwimean = ants.get_average_of_timeseries(dimg)
    dwireg = antspymm.tra_initializer(simg, dwimean, n_simulations=32, max_rotation=30, transform=['rigid'], verbose=True)
    dwimask = ants.apply_transforms(dwimean, simg_mask, dwireg['invtransforms'], whichtoinvert=[True], interpolator='nearestNeighbor')
    dwilab = ants.apply_transforms(dwimean, simg_labels, dwireg['invtransforms'], whichtoinvert=[True], interpolator='nearestNeighbor')
    dti = antspymm.dipy_dti_recon(dimg, dwibval, dwibvec, mask=dwimask)
    
    famask = ants.threshold_image(dti['FA'],0.01,1.0)
    stats_fa = ants.label_stats(dti['FA'], dwilab * famask )
    stats_md = ants.label_stats(dti['MD'], dwilab * ants.threshold_image(dti['MD'],1e-9,1.0))
    stats_fa = widen_summary_dataframe(stats_fa, description='LabelValue', value='Mean', skip_first=False )
    stats_md = widen_summary_dataframe(stats_md, description='LabelValue', value='Mean', skip_first=False )

    return {'registered_dwi': dwireg['warpedfixout'], 'dwimean': dwimean, 
        'dwimask': dwimask,
        'dwilab' :dwilab,
        'dti_results': dti, 
        'label_mean_fa': stats_fa,
        'label_mean_md': stats_md}

def rsfmri(rimg, simg, simg_mask):
    """
    Perform resting-state fMRI alignment.

    Parameters:
    rimg : ANTs image
        Resting-state fMRI image.
    simg : ANTs image
        Structural image.
    simg_mask : ANTs image
        Structural mask.
    simg_labels : ANTs image
        Structural labels within which we summarize features.

    Returns:
    dict
        Dictionary containing registered fMRI image and fMRI mask.
    """
    rsfmean = ants.get_average_of_timeseries(rimg)
    rsfreg = antspymm.tra_initializer(simg, rsfmean, n_simulations=32, max_rotation=30, transform=['rigid', 'syn'], verbose=True)
    rsfmask = ants.apply_transforms(rsfmean, simg_mask, rsfreg['invtransforms'], whichtoinvert=[True], interpolator='nearestNeighbor')
    
    return {'registered_rsfmri': rsfreg['warpedfixout'], 'rsfmri_mask': rsfmask}
