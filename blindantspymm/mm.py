
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
    if skip_first:
        mydf = mydf.drop(mydf.index[0])

    # Pivot table to convert long format to wide format, ensuring a single-row result
    df_wide = mydf.pivot_table(index=None, columns=description, values=value, aggfunc='first')

    # Rename columns to include the description prefix
    df_wide.columns = [f"{description}{col}.{value}" for col in df_wide.columns]

    # Reset index to ensure it's a proper DataFrame
    df_wide = df_wide.reset_index(drop=True)


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
        'label_geometry': label_geometry_measures_w,
        'brain_mask_overlap':mydice
    }

def perfusion( fmri, simg, simg_mask, simg_labels,
                   FD_threshold=0.5,
                   spa = (0., 0., 0., 0.),
                   nc = 3,
                   type_of_transform='Rigid',
                   tc='alternating',
                   n_to_trim=0,
                   m0_image = None,
                   m0_indices=None,
                   outlier_threshold=0.250,
                   add_FD_to_nuisance=False,
                   segment_timeseries=False,
                   trim_the_mask=4.25,
                   upsample=False,
                   perfusion_regression_model='linear',
                   verbose=False ):
  """
  Estimate perfusion from a BOLD time series image.  Will attempt to figure out the T-C labels from the data.  The function uses defaults to quantify CBF but these will usually not be correct for your own data.  See the function calculate_CBF for an example of how one might do quantification based on the outputs of this function specifically the perfusion, m0 and mask images that are part of the output dictionary.

  Arguments
  ---------
  fmri : BOLD fmri antsImage

  simg : ANTsImage
    input 3-D structural image (not brain extracted)

  simg_mask : ANTsImage
    input 3-D structural brain extraction

  simg_labels : ANTsImage
    structural segmentation or parcellation labels

  spa : gaussian smoothing for spatial and temporal component e.g. (1,1,1,0) in physical space coordinates

  nc  : number of components for compcor filtering

  type_of_transform : SyN or Rigid

  tc: string either alternating or split (default is alternating ie CTCTCT; split is CCCCTTTT)

  n_to_trim: number of volumes to trim off the front of the time series to account for initial magnetic saturation effects or to allow the signal to reach a steady state. in some cases, trailing volumes or other outlier volumes may need to be rejected.  this code does not currently handle that issue.

  m0_image: a pre-defined m0 image - we expect this to be 3D.  if it is not, we naively 
    average over the 4th dimension.

  m0_indices: which indices in the perfusion image are the m0.  if set, n_to_trim will be ignored.

  outlier_threshold (numeric): between zero (remove all) and one (remove none); automatically calculates outlierness and uses it to censor the time series.

  add_FD_to_nuisance: boolean

  segment_timeseries : boolean

  trim_the_mask : float >= 0 post-hoc method for trimming the mask

  upsample: boolean

  perfusion_regression_model: string 'linear', 'ransac', 'theilsen', 'huber', 'quantile', 'sgd'; 'linear' and 'huber' are the only ones that work ok by default and are relatively quick to compute.

  verbose : boolean

  Returns
  ---------
  a dictionary containing the derived network maps

  """
  import numpy as np
  import pandas as pd
  import re
  import math
  from sklearn.linear_model import RANSACRegressor, TheilSenRegressor, HuberRegressor, QuantileRegressor, LinearRegression, SGDRegressor
  from sklearn.multioutput import MultiOutputRegressor
  from sklearn.preprocessing import StandardScaler

  # remove outlier volumes
  if segment_timeseries:
    lo_vs_high = antspymm.segment_timeseries_by_meanvalue(fmri)
    fmri = antspymm.remove_volumes_from_timeseries( fmri, lo_vs_high['lowermeans'] )

  if m0_image is not None:
    if m0_image.dimension == 4:
      m0_image = ants.get_average_of_timeseries( m0_image )

  def select_regression_model(regression_model, min_samples=10 ):
    if regression_model == 'sgd' :
      sgd_regressor = SGDRegressor(penalty='elasticnet', alpha=1e-5, l1_ratio=0.15, max_iter=20000, tol=1e-3, random_state=42)
      return sgd_regressor
    elif regression_model == 'ransac':
      ransac = RANSACRegressor(
            min_samples=0.8,
            max_trials=10,         # Maximum number of iterations
#            min_samples=min_samples, # Minimum number samples to be chosen as inliers in each iteration
#            stop_probability=0.80,  # Probability to stop the algorithm if a good subset is found
#            stop_n_inliers=40,      # Stop if this number of inliers is found
#            stop_score=0.8,         # Stop if the model score reaches this value
#            n_jobs=int(os.getenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")) # Use all available CPU cores for parallel processing
        )
      return ransac
    models = {
        'sgd':SGDRegressor,
        'ransac': RANSACRegressor,
        'theilsen': TheilSenRegressor,
        'huber': HuberRegressor,
        'quantile': QuantileRegressor
    }
    return models.get(regression_model.lower(), LinearRegression)()

  def replicate_list(user_list, target_size):
    # Calculate the number of times the list should be replicated
    replication_factor = target_size // len(user_list)
    # Replicate the list and handle any remaining elements
    replicated_list = user_list * replication_factor
    remaining_elements = target_size % len(user_list)
    replicated_list += user_list[:remaining_elements]
    return replicated_list

  def one_hot_encode(char_list):
    unique_chars = list(set(char_list))
    encoding_dict = {char: [1 if char == c else 0 for c in unique_chars] for char in unique_chars}
    encoded_matrix = np.array([encoding_dict[char] for char in char_list])
    return encoded_matrix
  
  A = np.zeros((1,1))
  fmri_template, hlinds = antspymm.loop_timeseries_censoring( fmri, 0.10 )
  fmri_template = ants.get_average_of_timeseries( fmri_template )
  del hlinds
  rig = ants.registration( fmri_template, simg, 'BOLDRigid' )
  bmask = ants.apply_transforms( fmri_template, simg_mask, rig['fwdtransforms'][0], interpolator='genericLabel' )
  if m0_indices is None:
    if n_to_trim is None:
        n_to_trim=0
    mytrim=n_to_trim
  else:
    mytrim = 0
  perf_total_sigma = 1.5
  corrmo = antspymm.timeseries_reg(
    fmri, fmri_template,
    type_of_transform=type_of_transform,
    total_sigma=perf_total_sigma,
    fdOffset=2.0,
    trim = mytrim,
    output_directory=None,
    verbose=verbose,
    syn_metric='cc',
    syn_sampling=2,
    reg_iterations=[40,20,5] )
  if verbose:
      print("End rsfmri motion correction")

  if m0_image is not None:
      m0 = m0_image
  elif m0_indices is not None:
    not_m0 = list( range( fmri.shape[3] ) )
    not_m0 = [x for x in not_m0 if x not in m0_indices]
    if verbose:
        print( m0_indices )
        print( not_m0 )
    # then remove it from the time series
    m0 = antspymm.remove_volumes_from_timeseries( corrmo['motion_corrected'], not_m0 )
    m0 = ants.get_average_of_timeseries( m0 )
    corrmo['motion_corrected'] = antspymm.remove_volumes_from_timeseries( 
        corrmo['motion_corrected'], m0_indices )
    corrmo['FD'] = antspymm.remove_elements_from_numpy_array( corrmo['FD'], m0_indices )
    fmri = antspymm.remove_volumes_from_timeseries( fmri, m0_indices )

  ntp = corrmo['motion_corrected'].shape[3]
  if tc == 'alternating':
      tclist = replicate_list( ['C','T'], ntp )
  else:
      tclist = replicate_list( ['C'], int(ntp/2) ) + replicate_list( ['T'],  int(ntp/2) )

  tclist = one_hot_encode( tclist[0:ntp ] )
  fmrimotcorr=corrmo['motion_corrected']
  if outlier_threshold < 1.0 and outlier_threshold > 0.0:
    fmrimotcorr, hlinds = antspymm.loop_timeseries_censoring( fmrimotcorr, outlier_threshold, mask=None, verbose=verbose )
    tclist = antspymm.remove_elements_from_numpy_array( tclist, hlinds)
    corrmo['FD'] = antspymm.remove_elements_from_numpy_array( corrmo['FD'], hlinds )

  # redo template and registration at (potentially) upsampled scale
  fmri_template = ants.iMath( ants.get_average_of_timeseries( fmrimotcorr ), "Normalize" )
  if upsample:
      spc = ants.get_spacing( fmri )
      minspc = 2.0
      if min(spc[0:3]) < minspc:
          minspc = min(spc[0:3])
      newspc = [minspc,minspc,minspc]
      fmri_template = ants.resample_image( fmri_template, newspc, interp_type=0 )

  rig = ants.registration( fmri_template, simg, 'BOLDRigid' )
  bmask = ants.apply_transforms( fmri_template, 
    simg_mask, 
    rig['fwdtransforms'][0], 
    interpolator='genericLabel' )
  corrmo = antspymm.timeseries_reg(
        fmri, fmri_template,
        type_of_transform=type_of_transform,
        total_sigma=perf_total_sigma,
        fdOffset=2.0,
        trim = mytrim,
        output_directory=None,
        verbose=verbose,
        syn_metric='cc',
        syn_sampling=2,
        reg_iterations=[40,20,5] )
  if verbose:
        print("End 2nd rsfmri motion correction")

  if outlier_threshold < 1.0 and outlier_threshold > 0.0:
    corrmo['motion_corrected'] = antspymm.remove_volumes_from_timeseries( corrmo['motion_corrected'], hlinds )
    corrmo['FD'] = antspymm.remove_elements_from_numpy_array( corrmo['FD'], hlinds )

  regression_mask = bmask.clone()
  mytsnr = antspymm.tsnr( corrmo['motion_corrected'], bmask )
  mytsnrThresh = np.quantile( mytsnr.numpy(), 0.995 )
  tsnrmask = ants.threshold_image( mytsnr, 0, mytsnrThresh ).morphology("close",3)
  bmask = bmask * ants.iMath( tsnrmask, "FillHoles" )
  fmrimotcorr=corrmo['motion_corrected']
  und = fmri_template * bmask
  t1 = simg * simg_mask
  t1reg = ants.registration( und, t1, "SyNBold" )
  compcorquantile=0.50
  mycompcor = ants.compcor( fmrimotcorr,
    ncompcor=nc, quantile=compcorquantile, mask = bmask,
    filter_type='polynomial', degree=2 )
  tr = ants.get_spacing( fmrimotcorr )[3]
  simg = ants.smooth_image(fmrimotcorr, spa, sigma_in_physical_coordinates = True )
  nuisance = mycompcor['basis']
  nuisance = np.c_[ nuisance, mycompcor['components'] ]
  if add_FD_to_nuisance:
    nuisance = np.c_[ nuisance, corrmo['FD'] ]
  if verbose:
    print("make sure nuisance is independent of TC")
  nuisance = ants.regress_components( nuisance, tclist )
  regression_mask = bmask.clone()
  gmmat = ants.timeseries_to_matrix( simg, regression_mask )
  regvars = np.hstack( (nuisance, tclist ))
  coefind = regvars.shape[1]-1
  regvars = regvars[:,range(coefind)]
  predictor_of_interest_idx = regvars.shape[1]-1
  valid_perf_models = ['huber','quantile','theilsen','ransac', 'sgd', 'linear','SM']
  if verbose:
    print( "begin perfusion estimation with " + perfusion_regression_model + " model " )
  if perfusion_regression_model == 'linear':
    regression_model = LinearRegression()
    regression_model.fit( regvars, gmmat )
    coefind = regression_model.coef_.shape[1]-1
    perfimg = ants.make_image( regression_mask, regression_model.coef_[:,coefind] )
  elif perfusion_regression_model == 'SM': #
    import statsmodels.api as sm
    coeffs = np.zeros( gmmat.shape[1] )
    # Loop over each outcome column in the outcomes matrix
    for outcome_idx in range(gmmat.shape[1]):
        outcome = gmmat[:, outcome_idx]  # Select one outcome column
        model = sm.RLM(outcome, sm.add_constant(regvars), M=sm.robust.norms.HuberT())  # Huber's T norm for robust regression
        results = model.fit()
        coefficients = results.params  # Coefficients of all predictors
        coeffs[outcome_idx] = coefficients[predictor_of_interest_idx]
    perfimg = ants.make_image( regression_mask, coeffs )
  elif perfusion_regression_model in valid_perf_models :
    scaler = StandardScaler()
    gmmat = scaler.fit_transform(gmmat)
    coeffs = np.zeros( gmmat.shape[1] )
    huber_regressor = select_regression_model( perfusion_regression_model )
    multioutput_model = MultiOutputRegressor(huber_regressor)
    multioutput_model.fit( regvars, gmmat )
    ct=0
    for i, estimator in enumerate(multioutput_model.estimators_):
      coefficients = estimator.coef_
      coeffs[ct]=coefficients[predictor_of_interest_idx]
      ct=ct+1
    perfimg = ants.make_image( regression_mask, coeffs )
  else:
    raise ValueError( perfusion_regression_model + " regression model is not found.")
  negative_voxels = ( perfimg < 0.0 ).sum() / np.prod( perfimg.shape ) * 100.0
  if negative_voxels > 0.5:
    perfimg = perfimg * (-1.0)
    negative_voxels = ( perfimg < 0.0 ).sum() / np.prod( perfimg.shape ) * 100.0
  perfimg[ perfimg < 0.0 ] = 0.0 # non-physiological

  # LaTeX code for Cerebral Blood Flow (CBF) calculation using ASL MRI
  """
CBF = \\frac{\\Delta M \\cdot \\lambda}{2 \\cdot T_1 \\cdot \\alpha \\cdot M_0 \\cdot (e^{-\\frac{w}{T_1}} - e^{-\\frac{w + \\tau}{T_1}})}

Where:
- \\Delta M is the difference in magnetization between labeled and control images.
- \\lambda is the brain-blood partition coefficient, typically around 0.9 mL/g.
- T_1 is the longitudinal relaxation time of blood, which is a tissue-specific constant.
- \\alpha is the labeling efficiency.
- M_0 is the equilibrium magnetization of brain tissue (from the M0 image).
- w is the post-labeling delay, the time between the end of the labeling and the acquisition of the image.
- \\tau is the labeling duration.
  """
  if m0_indices is None and m0_image is None:
    m0 = ants.get_average_of_timeseries( fmrimotcorr )
  else:
    # register m0 to current template
    m0reg = ants.registration( fmri_template, m0, 'Rigid', verbose=False )
    m0 = m0reg['warpedmovout']

  if ntp == 2 :
      img0 = ants.slice_image( corrmo['motion_corrected'], axis=3, idx=0 )
      img1 = ants.slice_image( corrmo['motion_corrected'], axis=3, idx=1 )
      if m0_image is None:
        if img0.mean() < img1.mean():
            perfimg=img0
            m0=img1
        else:
            perfimg=img1
            m0=img0
      else:
        if img0.mean() < img1.mean():
            perfimg=img1-img0
        else:
            perfimg=img0-img1

  cbf = antspymm.calculate_CBF(
      Delta_M=perfimg, M_0=m0, mask=bmask )
  if trim_the_mask > 0.0 and False :
    bmask = antspymm.trim_dti_mask( cbf, bmask, trim_the_mask )
    perfimg = perfimg * bmask
    cbf = cbf * bmask

  if verbose:
    print("perfimg.max() " + str(  perfimg.max() ) )
  outdict = {}
  outdict['meanBold'] = und
  outdict['brainmask'] = bmask
  rsfNuisance = pd.DataFrame( nuisance )
  rsfNuisance['FD']=corrmo['FD']

  if verbose:
      print("perfusion dataframe begin")
  dktseg = ants.apply_transforms( und, simg_labels,
    t1reg['fwdtransforms'], interpolator = 'genericLabel' ) * bmask

  stats_pf = ants.label_stats(perfimg,dktseg )
  stats_cb = ants.label_stats(cbf,dktseg )
  stats_pf = widen_summary_dataframe(stats_pf, description='LabelValue', value='Mean', skip_first=True )
  stats_cb = widen_summary_dataframe(stats_cb, description='LabelValue', value='Mean', skip_first=True )

  stats_pf = pd.concat( [stats_pf,stats_cb], axis=1, ignore_index=False )
  if verbose:
      print("perfusion dataframe end")

  outdict['perfusion']=perfimg
  outdict['cbf']=cbf
  outdict['m0']=m0
  outdict['label_mean']=stats_pf
  outdict['motion_corrected'] = corrmo['motion_corrected']
  outdict['brain_mask'] = bmask
  outdict['nuisance'] = rsfNuisance
  outdict['tsnr'] = mytsnr
#  outdict['ssnr'] = antspymm.slice_snr( corrmo['motion_corrected'], csfAndWM, gmseg )
  outdict['dvars'] = antspymm.dvars( corrmo['motion_corrected'], bmask )
  outdict['high_motion_count'] = (rsfNuisance['FD'] > FD_threshold ).sum()
  outdict['high_motion_pct'] = (rsfNuisance['FD'] > FD_threshold ).sum() / rsfNuisance.shape[0]
  outdict['FD_max'] = rsfNuisance['FD'].max()
  outdict['FD_mean'] = rsfNuisance['FD'].mean()
  outdict['FD_sd'] = rsfNuisance['FD'].std()
  outdict['bold_evr'] =  antspyt1w.patch_eigenvalue_ratio( und, 512, [16,16,16], evdepth = 0.9, mask = bmask )
  outdict['t1reg'] = t1reg
  outdict['outlier_volumes']=hlinds
  outdict['n_outliers']=len(hlinds)
  outdict['negative_voxels']=negative_voxels
  return antspymm.convert_np_in_dict( outdict )

    
def pet( pet3d, simg, simg_mask, simg_labels,
                   spa = (0., 0., 0.),
                   type_of_transform='Rigid',
                   upsample=True,
                   verbose=False ):
  """
  Estimate perfusion from a BOLD time series image.  Will attempt to figure out the T-C labels from the data.  The function uses defaults to quantify CBF but these will usually not be correct for your own data.  See the function calculate_CBF for an example of how one might do quantification based on the outputs of this function specifically the perfusion, m0 and mask images that are part of the output dictionary.

  Arguments
  ---------
  pet3d : 3D PET antsImage

  simg : ANTsImage
    input 3-D structural image (not brain extracted)

  simg_mask : ANTsImage
    input 3-D structural brain extraction

  simg_labels : ANTsImage
    structural segmentation or parcellation labels

  type_of_transform : SyN or Rigid

  upsample: boolean

  verbose : boolean

  Returns
  ---------
  a dictionary containing the derived summary data

  """
  import numpy as np
  import pandas as pd
  import re
  import math

  pet3dr=pet3d
  if upsample:
      spc = ants.get_spacing( pet3d )
      minspc = 1.0
      if min(spc) < minspc:
        minspc = min(spc)
      newspc = [minspc,minspc,minspc]
      pet3dr = ants.resample_image( pet3d, newspc, interp_type=0 )

  rig = ants.registration( pet3dr, simg, 'BOLDRigid' )
  bmask = ants.apply_transforms( pet3dr, 
    simg_mask, 
    rig['fwdtransforms'][0], 
    interpolator='genericLabel' )
  if verbose:
    print("End structural=>pet registration")

  und = pet3dr * bmask
  t1reg = rig # ants.registration( und, t1, "Rigid" )
  if verbose:
    print("pet3dr.max() " + str(  pet3dr.max() ) )
  if verbose:
      print("pet3d dataframe begin")
  dktseg = ants.apply_transforms( und, simg_labels,
    t1reg['fwdtransforms'], interpolator = 'genericLabel' ) * bmask
  df_pet3d = ants.label_stats( und, dktseg )
  stats_pet = widen_summary_dataframe(df_pet3d, description='LabelValue', value='Mean', skip_first=True )
  if verbose:
      print("pet3d dataframe end")
      print( stats_pet )

  outdict = {}
  outdict['pet3d'] = pet3dr
  outdict['brainmask'] = bmask
  outdict['labels'] = dktseg
  outdict['t1reg'] = t1reg
  outdict['label_mean']=stats_pet
  return antspymm.convert_np_in_dict( outdict )


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
    Perform resting-state fMRI processing.  Connectivity and activation measurements.

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
        Dictionary containing processed resting state fMRI data.
    """
    rsfmean = ants.get_average_of_timeseries(rimg)
    rsfreg = antspymm.tra_initializer(simg, rsfmean, n_simulations=32, max_rotation=30, transform=['rigid', 'syn'], verbose=True)
    rsfmask = ants.apply_transforms(rsfmean, simg_mask, rsfreg['invtransforms'], whichtoinvert=[True], interpolator='nearestNeighbor')



    return {'registered_rsfmri': rsfreg['warpedfixout'], 'rsfmri_mask': rsfmask}
