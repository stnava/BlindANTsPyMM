
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

