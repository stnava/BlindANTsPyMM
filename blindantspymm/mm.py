
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
    reg = ants.registration(
        fixed=template,
        moving=simg,
        type_of_transform=type_of_transform,
        mask=template_mask
    )
    
    # Warp the subject mask to the template space
    warped_mask = ants.apply_transforms(
        fixed=template,
        moving=simg_mask,
        transformlist=reg['fwdtransforms'],
        interpolator='nearestNeighbor'
    )
    
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
        transform=reg['fwdtransforms'][0],
        do_log=True
    )
    
    # Compute label geometry measures
    label_geometry_measures = ants.label_geometry_measures(inverse_warped_labels)
    
    return {
        'warped_image': reg['warpedmovout'],
        'warped_mask': warped_mask,
        'inverse_warped_labels': inverse_warped_labels,
        'forward_transforms': reg['fwdtransforms'],
        'inverse_transforms': reg['invtransforms'],
        'jacobian': jacobian,
        'label_geometry_measures': label_geometry_measures
    }



def dwi():
    return None

def rsfmri():
    return None
    
def pet():
    return None
    
