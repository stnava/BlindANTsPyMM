
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


def validate_registration_strategy(registration_strategy):
    """
    Validates the registration strategy list.

    Parameters:
    registration_strategy : list of str
        A list where:
        - The first entry must be 'brain_based' or 'head_based' and refers to the features used for the fixed and moving images
        - The second entry must be 'simple', 'standard' or 'optimal' and corresponds to the registration strategy that is used
        - The third entry must be 'normalize' or 'rank' and corresponds to the intensity transformations applied to the fixed and moving image

    Raises:
    ValueError
        If the input is not valid.
    """
    valid_first = {'brain_based', 'head_based'}
    valid_second = {'simple', 'standard', 'optimal'}
    valid_third = {'normalize', 'rank'}
    
    if not isinstance(registration_strategy, list) or len(registration_strategy) != 3:
        raise ValueError("registration_strategy must be a list of three elements.")
    
    first, second, third = registration_strategy
    
    if first not in valid_first:
        raise ValueError(f"Invalid first entry: {first}. Must be one of {valid_first}.")
    if second not in valid_second:
        raise ValueError(f"Invalid second entry: {second}. Must be one of {valid_second}.")
    if third not in valid_third:
        raise ValueError(f"Invalid third entry: {third}. Must be one of {valid_third}.")


def widen_summary_dataframe(mydf, description='Label', value='VolumeInMillimeters', skip_first=False, prefix='' ):
    """
    Convert a long-format dataframe to a single-row wide-format dataframe.

    Parameters:
    - mydf: pd.DataFrame, long-format dataframe with at least two columns: 
      one categorical identifier (e.g., 'Label') and one value column.
    - description: str, column name that represents the categories (e.g., 'Label').
    - value: str, column name with numerical values to be spread into wide format.
    - skip_first: boolean, skips the first row
    - prefix: string prefix for column names

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

    df_wide = df_wide.add_prefix( prefix )

    return df_wide


def structural(simg, simg_mask, template, template_mask, template_labels, type_of_transform='antsRegistrationSyNQuick[s]', upsample=0.0 ):
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
    upsample: spacing to which we upsample the image (uniform); set to zero to skip

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
    if upsample > 0.:
        spc = ants.get_spacing( simg )
        minspc = upsample
        newspc = [minspc]*simg.dimension
        simg = ants.resample_image( simg, newspc, interp_type=0 )
        simg_mask = ants.resample_image( simg_mask, newspc, interp_type=1 )
    sbrain = simg * simg_mask
    sbrain[ sbrain < 0.0 ] = 0.0
    ritbrain=ants.rank_intensity(tbrain)
    risbrain=ants.rank_intensity(sbrain)
    reg = ants.registration(
        fixed=ritbrain,
        moving=risbrain,
        type_of_transform=type_of_transform
    )
    #######
    intermodality_similarity = ants.image_mutual_information( ritbrain, reg['warpedmovout'] )
    #######
    inverse_warped_mask = ants.apply_transforms(
        fixed=simg,
        moving=template_mask,
        transformlist=reg['invtransforms'],
        interpolator='genericLabel'
    )

    # for QC purposes
    mydice = ants.label_overlap_measures( inverse_warped_mask, simg_mask )

    # Warp the template labels to subject space using the inverse transform
    inverse_warped_labels = ants.apply_transforms(
        fixed=simg,
        moving=template_labels,
        transformlist=reg['invtransforms'],
        interpolator='genericLabel'
    )
    
    # Compute the Jacobian determinant of the transformation
    jacobian = ants.create_jacobian_determinant_image(
        domain_image=template,
        tx=reg['fwdtransforms'][0],
        do_log=True
    )
    
    # Compute label geometry measures
    label_geometry_measures = ants.label_geometry_measures(inverse_warped_labels)
    label_geometry_measures_w = widen_summary_dataframe( label_geometry_measures, value='VolumeInMillimeters' )
    label_geometry_measures_w = pd.concat( 
        [   label_geometry_measures_w,
            widen_summary_dataframe( label_geometry_measures, 
                value='SurfaceAreaInMillimetersSquared') ], axis=1 )
    label_geometry_measures_w=label_geometry_measures_w.add_prefix("smri.")
    outdict = {
        'brainmask': inverse_warped_mask,
        'labels': inverse_warped_labels,
        'registration_result': reg,
        'jacobian': jacobian,
        'label_geometry': label_geometry_measures_w,
        'brain_mask_overlap':mydice,
        'intermodality_similarity': intermodality_similarity
    }
    return antspymm.convert_np_in_dict( outdict )

def perfusion( fmri, simg, simg_mask, simg_labels,
                   FD_threshold=0.5,
                   spa = (0., 0., 0., 0.),
                   nc = 3,
                   type_of_intermodality_transform='SyN',
                   type_of_motion_transform='Rigid',
                   tc='alternating',
                   n_to_trim=0,
                   m0_image = None,
                   m0_indices=None,
                   outlier_threshold=0.,
                   add_FD_to_nuisance=False,
                   segment_timeseries=False,
                   trim_the_mask=4.25,
                   upsample=2.0,
                   perfusion_regression_model='linear',
                   registration_strategy=['head_based','standard','rank'],
                   verbose=False ):
  validate_registration_strategy( registration_strategy )
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

  type_of_intermodality_transform : SyN or Rigid for mapping structure to this modality

  type_of_motion_transform : SyN or Rigid for motion correction

  tc: string either alternating or split (default is alternating ie CTCTCT; split is CCCCTTTT)

  n_to_trim: number of volumes to trim off the front of the time series to account for initial magnetic saturation effects or to allow the signal to reach a steady state. in some cases, trailing volumes or other outlier volumes may need to be rejected.  this code does not currently handle that issue.

  m0_image: a pre-defined m0 image - we expect this to be 3D.  if it is not, we naively 
    average over the 4th dimension.

  m0_indices: which indices in the perfusion image are the m0.  if set, n_to_trim will be ignored.

  outlier_threshold (numeric): between zero (remove all) and one (remove none); automatically calculates outlierness and uses it to censor the time series.

  add_FD_to_nuisance: boolean

  segment_timeseries : boolean

  trim_the_mask : float >= 0 post-hoc method for trimming the mask

  upsample: spacing to which we upsample the image (uniform); set to zero to skip

  perfusion_regression_model: string 'linear', 'ransac', 'theilsen', 'huber', 'quantile', 'sgd'; 'linear' and 'huber' are the only ones that work ok by default and are relatively quick to compute.

  registration_strategy : a list of strings with the first entry being one of 'brain_based', 'head_based', the 2nd being 'simple', 'standard', or 'optimal', the 3rd being 'normalize' or 'rank'; see help for validate_registration_strategy

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

  validate_registration_strategy( registration_strategy )


  if verbose:
    print("rsf-perfusion registration")
  A = np.zeros((1,1))
  if upsample > 0.:
      spc = ants.get_spacing( fmri )
      minspc = upsample
      if min(spc[0:3]) < minspc:
          minspc = min(spc[0:3])
      newspc = [minspc,minspc,minspc, spc[3]]
      fmri = ants.resample_image( fmri, newspc, interp_type=0 )
  fmri_template = antspymm.get_average_rsf( fmri )
  simg_fgd = simg * ants.threshold_image( simg, 'Otsu', 1)
  if registration_strategy[0] == 'brain_based' :
    simg_fgd = simg * simg_mask
  rig, intermodality_similarity = reg_opt( simg_fgd, fmri_template, transform_list=[ 'Rigid' ], 
    registration_strategy = registration_strategy[1], intensity_transform=registration_strategy[2] )
  if verbose:
    print("rsf-perfusion template mask and labels")
  bmask = ants.apply_transforms( fmri_template, simg_mask, rig['invtransforms'], interpolator='genericLabel', which_to_invert=[True,False] )
  rsflabels = ants.apply_transforms( fmri_template, simg_labels, rig['invtransforms'], interpolator='genericLabel', which_to_invert=[True,False] )
#  if verbose:
#    ants.plot( fmri_template, rsflabels, crop=True, axis=2 )
  if m0_indices is None:
    if n_to_trim is None:
        n_to_trim=0
    mytrim=n_to_trim
  else:
    mytrim = 0
  perf_total_sigma = 1.5
  corrmo = antspymm.timeseries_reg(
    fmri, fmri_template,
    type_of_transform=type_of_motion_transform,
    total_sigma=perf_total_sigma,
    fdOffset=2.0,
    trim = mytrim,
    output_directory=None,
    verbose=verbose,
    syn_metric='cc',
    syn_sampling=2,
    reg_iterations=[40,20,5] )
  if verbose:
      print("End rsfmri-perfusion motion correction")

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

  corrmo = antspymm.timeseries_reg(
        fmri, fmri_template,
        type_of_transform=type_of_motion_transform,
        total_sigma=perf_total_sigma,
        fdOffset=2.0,
        trim = mytrim,
        output_directory=None,
        verbose=verbose,
        syn_metric='cc',
        syn_sampling=2,
        reg_iterations=[40,20,5] )
  if verbose:
        print("End 2nd rsfmri-perfusion motion correction")

  if outlier_threshold < 1.0 and outlier_threshold > 0.0:
    corrmo['motion_corrected'] = antspymm.remove_volumes_from_timeseries( corrmo['motion_corrected'], hlinds )
    corrmo['FD'] = antspymm.remove_elements_from_numpy_array( corrmo['FD'], hlinds )

  regression_mask = bmask.clone()
  mytsnr = antspymm.tsnr( corrmo['motion_corrected'], bmask )
  # mytsnrThresh = np.quantile( mytsnr.numpy(), 0.995 )
  # tsnrmask = ants.threshold_image( mytsnr, 0, mytsnrThresh ).morphology("close",3)
  # bmask = bmask * ants.iMath( tsnrmask, "FillHoles" )
  fmrimotcorr=corrmo['motion_corrected']
  compcorquantile=0.90
  mycompcor = ants.compcor( fmrimotcorr,
    ncompcor=nc, quantile=compcorquantile, mask = bmask,
    filter_type='polynomial', degree=2 )
  tr = ants.get_spacing( fmrimotcorr )[3]
  simg = ants.smooth_image(fmrimotcorr, spa, sigma_in_physical_coordinates = True )
  nuisance = mycompcor['basis']
  nuisance = np.c_[ nuisance, mycompcor['components'] ]
  nuisance = mycompcor['components']
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
  if verbose:
      print("perfusion dataframe begin")
  stats_pf = ants.label_stats(perfimg,rsflabels )
  stats_cb = ants.label_stats(cbf,rsflabels )
  stats_pf = widen_summary_dataframe(stats_pf, description='LabelValue', value='Mean', skip_first=True, prefix='asl.perf.' )
  stats_cb = widen_summary_dataframe(stats_cb, description='LabelValue', value='Mean', skip_first=True, prefix='asl.cbf.' )
  stats_pf = pd.concat( [stats_pf,stats_cb], axis=1, ignore_index=False )
  
  if verbose:
      print("perfusion dataframe end")

  outdict = {}
  outdict['meanBold'] = fmri_template
  outdict['brainmask'] = bmask
  outdict['labels'] = rsflabels
  rsfNuisance = pd.DataFrame( nuisance )
  if not add_FD_to_nuisance:
    rsfNuisance['FD']=corrmo['FD']
  outdict['perfusion']=perfimg
  outdict['cbf']=cbf
  outdict['m0']=m0
  outdict['label_mean']=stats_pf
  outdict['motion_corrected'] = corrmo['motion_corrected']
  outdict['nuisance'] = rsfNuisance
  outdict['tsnr'] = mytsnr
#  outdict['ssnr'] = antspymm.slice_snr( corrmo['motion_corrected'], csfAndWM, gmseg )
  outdict['dvars'] = antspymm.dvars( corrmo['motion_corrected'], bmask )
  outdict['high_motion_count'] = (rsfNuisance['FD'] > FD_threshold ).sum()
  outdict['high_motion_pct'] = (rsfNuisance['FD'] > FD_threshold ).sum() / rsfNuisance.shape[0]
  outdict['FD_max'] = rsfNuisance['FD'].max()
  outdict['FD_mean'] = rsfNuisance['FD'].mean()
  outdict['FD_sd'] = rsfNuisance['FD'].std()
  outdict['bold_evr'] = antspyt1w.patch_eigenvalue_ratio( fmri_template, 512,
    [16,16,16], evdepth = 0.9, mask = bmask )
  outdict['negative_voxels']=negative_voxels
  outdict['registration_result']=rig
  outdict['intermodality_similarity'] = intermodality_similarity  
  return antspymm.convert_np_in_dict( outdict )


def pet( pet3d, simg, simg_mask, simg_labels,
                   spa = (0., 0., 0.),
                   type_of_transform='BOLDRigid',
                   upsample=1.,
                   reorient=True,
                   registration_strategy=['head_based','simple','normalize'],
                   verbose=False ):
  validate_registration_strategy( registration_strategy )
  """
  Summarize PET data by a (registration) transform of structural labels to the PET space.

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

  upsample: spacing to which we upsample the image (set to zero to skip)

  reorient: boolean to allow reorientation

  registration_strategy : a list of strings with the first entry being one of 'brain_based', 'head_based', the 2nd being 'simple', 'standard', or 'optimal', the 3rd being 'normalize' or 'rank'; see help for validate_registration_strategy

  verbose : boolean

  Returns
  ---------
  a dictionary containing the derived summary data

  """
  import numpy as np
  import pandas as pd
  import re
  import math
  pet3dr=ants.image_clone(pet3d)
  if reorient:
    # take care of the (potential) orientation issues
    pet3dr=ants.reorient_image2( pet3d, ants.get_orientation(simg) )
  if upsample > 0.0:
      spc = ants.get_spacing( pet3dr )
      minspc = upsample
      if min(spc) < minspc:
        minspc = min(spc)
      newspc = [minspc,minspc,minspc]
      pet3dr = ants.resample_image( pet3dr, newspc, interp_type=0 )

  if verbose:
    print("pet registration")
  if registration_strategy[0] == 'brain_based' :
    simg_fgd = simg * simg_mask
  else:
    simg_fgd = simg
  rig, intermodality_similarity = reg_opt( simg_fgd, pet3dr, transform_list=[ 'Rigid' ], 
    registration_strategy = registration_strategy[1], intensity_transform=registration_strategy[2] )
  if verbose:
    print("pet mask and labels")
  bmask = ants.apply_transforms( pet3dr, simg_mask, rig['invtransforms'], interpolator='genericLabel', which_to_invert=[True,False] )
  petlabels = ants.apply_transforms( pet3dr, simg_labels, rig['invtransforms'], interpolator='genericLabel', which_to_invert=[True,False] )
  if verbose:
    print("End structural=>pet registration")

  und = pet3dr * bmask
  if verbose:
    print("pet3dr.max() " + str(  und.max() ) )
  if verbose:
      print("pet3d dataframe begin")
  df_pet3d = ants.label_stats( und, petlabels )
  stats_pet = widen_summary_dataframe(df_pet3d, description='LabelValue', value='Mean', skip_first=True, prefix='pet.' )
  if verbose:
      print("pet3d dataframe end")
      print( stats_pet )

  outdict = {}
  outdict['pet_resam'] = pet3dr
  outdict['brainmask'] = bmask
  outdict['labels'] = petlabels
  outdict['registration_result'] = rig
  outdict['label_mean']=stats_pet
  outdict['intermodality_similarity'] = intermodality_similarity
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
    prior_templatelab = ants.apply_transforms(timgbc, prior_templatelab0, prior_templatereg['fwdtransforms'], interpolator='genericLabel')
    prior_templatelab_og = ants.apply_transforms(template, prior_templatelab0, prior_templatereg['fwdtransforms'], interpolator='genericLabel')
    return {'transformed_template': timgbc, 
        'transformed_labels': prior_templatelab, 
        'transformed_labels_orig_space': prior_templatelab_og, 
        'registration_result' : prior_templatereg }


def dwi(dimg, simg, simg_mask, simg_labels, dwibval, dwibvec, upsample=0.,                   registration_strategy=['head_based','simple','normalize'] ):
    validate_registration_strategy( registration_strategy )
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
    upsample : float optionally isotropically upsample data to upsample (the parameter value) in mm during the registration process if data is below that resolution; if the input spacing is less than that provided by the user, the data will simply be resampled to isotropic resolution

    registration_strategy : a list of strings with the first entry being one of 'brain_based', 'head_based', the 2nd being 'simple', 'standard', or 'optimal', the 3rd being 'normalize' or 'rank'; see help for validate_registration_strategy

    Returns:
    dict
        Dictionary containing registered DWI image and diffusion tensor imaging (DTI) results.
    """
    if upsample > 0.:
        spc = ants.get_spacing( dimg )
        minspc = upsample
        if min(spc[0:3]) < upsample:
            minspc = min(spc[0:3])
        newspc = [minspc,minspc,minspc, spc[3]]
        dimg = ants.resample_image( dimg, newspc, interp_type=0 )
    dwimean = ants.get_average_of_timeseries(dimg)

    if registration_strategy[0] == 'brain_based' :
        simg_fgd = simg * simg_mask
    else:
        simg_fgd = simg
    dwireg, intermodality_similarity = reg_opt( simg_fgd, dwimean, transform_list=[ 'Rigid' ], 
        registration_strategy = registration_strategy[1], intensity_transform=registration_strategy[2] )
    dwimask = ants.apply_transforms(dwimean, simg_mask, dwireg['invtransforms'],
        interpolator='genericLabel')
    dwilab = ants.apply_transforms(dwimean, simg_labels, dwireg['invtransforms'],
        interpolator='genericLabel')
    dti = antspymm.dipy_dti_recon(dimg, dwibval, dwibvec, mask=dwimask)    
    famask = ants.threshold_image(dti['FA'],0.01,1.0)
    stats_fa = ants.label_stats(dti['FA'], dwilab * famask )
    stats_md = ants.label_stats(dti['MD'], dwilab * ants.threshold_image(dti['MD'],1e-9,1.0))
    stats_fa = widen_summary_dataframe(stats_fa, description='LabelValue', value='Mean', skip_first=False, prefix='dwi.fa.' )
    stats_md = widen_summary_dataframe(stats_md, description='LabelValue', value='Mean', skip_first=False, prefix='dwi.md.' )
    outdict = {
        'dwimean': dwimean, 
        'brainmask': dwimask,
        'labels' : dwilab,
        'dti_results': dti, 
        'label_mean_fa': stats_fa,
        'label_mean_md': stats_md,
        'registration_result': dwireg,
        'intermodality_similarity': intermodality_similarity }
    return antspymm.convert_np_in_dict( outdict )


def rsfmri_to_correlation_matrix(rsfmri, roi_labels):
    """
    Compute the correlation matrix for an rs-fMRI image using the given ROI labels.
    
    Parameters:
    rsfmri (ants.ANTsImage): The resting-state fMRI 4D image.
    roi_labels (ants.ANTsImage): The ROI label image.
    
    Returns:
    np.ndarray: The correlation matrix of the extracted mean time series from each ROI.
    """
    unique_labels = np.unique(roi_labels.numpy())
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (label 0)
    
    meanROI = np.zeros((rsfmri.shape[-1], len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        roi_mask = roi_labels == label
        meanROI[:, i] = ants.timeseries_to_matrix(rsfmri, roi_mask).mean(axis=1)
    
    corMat = np.corrcoef(meanROI, rowvar=False)
    return corMat


def rsfmri_to_correlation_matrix_wide(rsfmri, roi_labels, corrprefix='rsf.'):
    """
    Compute the correlation matrix from an rsfMRI image given an ROI label image.
    
    Parameters:
    rsfmri : ANTsImage
        4D resting-state fMRI image.
    roi_labels : ANTsImage
        3D image with labeled ROIs.
    corrprefix: string
        prefix to prepend to column names

    Returns:
    pd.DataFrame
        A one-row dataframe containing the non-diagonal elements of the correlation matrix,
        named according to the ROI pairs.
    """
    import itertools
    # Get unique ROI labels (excluding background)
    unique_labels = np.unique(roi_labels.numpy())
    unique_labels = unique_labels[unique_labels > 0]

    # Compute mean time series for each ROI
    meanROI = np.zeros((rsfmri.shape[-1], len(unique_labels)))
    for i, label in enumerate(unique_labels):
        roi_mask = ants.threshold_image(roi_labels, label, label)
        meanROI[:, i] = ants.timeseries_to_matrix(rsfmri, roi_mask).mean(axis=1)

    # Compute correlation matrix
    corMat = np.corrcoef(meanROI, rowvar=False)

    # Extract upper triangle (excluding diagonal)
    roi_pairs = list(itertools.combinations(unique_labels, 2))
    cor_values = corMat[np.triu_indices(len(unique_labels), k=1)]

    # Construct a well-named dataframe without special characters
    col_names = [f"Corr_Label{int(a)}_Label{int(b)}" for a, b in roi_pairs]
    df_wide = pd.DataFrame([cor_values], columns=col_names)
    df_wide = df_wide.add_prefix( corrprefix )
    return df_wide

def rsfmri( fmri, simg, simg_mask, simg_labels,
    f=[0.03, 0.08],
    FD_threshold=5.0,
    spa = None,
    spt = None,
    nc = 5,
    outlier_threshold=0.250,
    ica_components = 0,
    impute = True,
    censor = True,
    despike = 0.0,
    motion_as_nuisance = True,
    powers = False,
    upsample = 3.0,
    clean_tmp = None,
    registration_strategy=['head_based','standard','rank'],
    verbose=False ):
  validate_registration_strategy( registration_strategy )
  """
  Compute resting state network correlation maps based on user input labels.

  registration - despike - anatomy - smooth - nuisance - bandpass - regress.nuisance - censor - falff - correlationss

  Arguments
  ---------
  fmri : BOLD fmri antsImage

  simg : ANTs image
        Structural image.

  simg_mask : ANTs image
        Structural mask.

  simg_labels : ANTs image
        Structural labels within which we summarize features.

  f : band pass limits for frequency filtering; we use high-pass here as per Shirer 2015

  spa : gaussian smoothing for spatial component (physical coordinates)

  spt : gaussian smoothing for temporal component

  nc  : number of components for compcor filtering; if less than 1 we estimate on the fly based on explained variance; 10 wrt Shirer 2015 5 from csf and 5 from wm

  ica_components : integer if greater than 0 then include ica components

  impute : boolean if True, then use imputation in f/ALFF, PerAF calculation

  censor : boolean if True, then use censoring (censoring)

  despike : if this is greater than zero will run voxel-wise despiking in the 3dDespike (afni) sense; after motion-correction

  motion_as_nuisance: boolean will add motion and first derivative of motion as nuisance

  powers : boolean if True use Powers nodes otherwise 2023 Yeo 500 homotopic nodes (10.1016/j.neuroimage.2023.120010)

  upsample : float optionally isotropically upsample data to upsample (the parameter value) in mm during the registration process if data is below that resolution; if the input spacing is less than that provided by the user, the data will simply be resampled to isotropic resolution

  clean_tmp : will automatically try to clean the tmp directory - not recommended but can be used in distributed computing systems to help prevent failures due to accumulation of tmp files when doing large-scale processing.  if this is set, the float value clean_tmp will be interpreted as the age in hours of files to be cleaned.

  registration_strategy : a list of strings with the first entry being one of 'brain_based', 'head_based', the 2nd being 'simple', 'standard', or 'optimal', the 3rd being 'normalize' or 'rank'; see help for validate_registration_strategy

  verbose : boolean

  Returns
  ---------
  a dictionary containing the derived network maps

  References
  ---------

  10.1162/netn_a_00071 "Methods that included global signal regression were the most consistently effective de-noising strategies."

  10.1016/j.neuroimage.2019.116157 "frontal and default model networks are most reliable whereas subcortical neteworks are least reliable"  "the most comprehensive studies of pipeline effects on edge-level reliability have been done by shirer (2015) and Parkes (2018)" "slice timing correction has minimal impact" "use of low-pass or narrow filter (discarding  high frequency information) reduced both reliability and signal-noise separation"

  10.1016/j.neuroimage.2017.12.073: Our results indicate that (1) simple linear regression of regional fMRI time series against head motion parameters and WM/CSF signals (with or without expansion terms) is not sufficient to remove head motion artefacts; (2) aCompCor pipelines may only be viable in low-motion data; (3) volume censoring performs well at minimising motion-related artefact but a major benefit of this approach derives from the exclusion of high-motion individuals; (4) while not as effective as volume censoring, ICA-AROMA performed well across our benchmarks for relatively low cost in terms of data loss; (5) the addition of global signal regression improved the performance of nearly all pipelines on most benchmarks, but exacerbated the distance-dependence of correlations between motion and functional connec- tivity; and (6) group comparisons in functional connectivity between healthy controls and schizophrenia patients are highly dependent on preprocessing strategy. We offer some recommendations for best practice and outline simple analyses to facilitate transparent reporting of the degree to which a given set of findings may be affected by motion-related artefact.

  10.1016/j.dcn.2022.101087 : We found that: 1) the most efficacious pipeline for both noise removal and information recovery included censoring, GSR, bandpass filtering, and head motion parameter (HMP) regression, 2) ICA-AROMA performed similarly to HMP regression and did not obviate the need for censoring, 3) GSR had a minimal impact on connectome fingerprinting but improved ISC, and 4) the strictest censoring approaches reduced motion correlated edges but negatively impacted identifiability.

  """

  import warnings

  if clean_tmp is not None:
    clean_tmp_directory( age_hours = clean_tmp )

  if nc > 1:
    nc = int(nc)
  else:
    nc=float(nc)

  type_of_motion_transform='Rigid' # , # should probably not change this
  remove_it=True
  output_directory = tempfile.mkdtemp()
  output_directory_w = output_directory + "/ts_t1_reg/"
  os.makedirs(output_directory_w,exist_ok=True)
  ofnt1tx = tempfile.NamedTemporaryFile(delete=False,suffix='t1_deformation',dir=output_directory_w).name

  import numpy as np
# Assuming core and utils are modules or packages with necessary functions
  if verbose:
    print("rsf template")
  if upsample > 0.:
      spc = ants.get_spacing( fmri )
      minspc = upsample
      if min(spc[0:3]) < upsample:
          minspc = min(spc[0:3])
      newspc = [minspc,minspc,minspc, spc[3]]
      fmri = ants.resample_image( fmri, newspc, interp_type=0 )
  fmri_template = antspymm.get_average_rsf( fmri )
  simg_fgd = simg * ants.threshold_image( simg, 'Otsu', 1)
  if registration_strategy[0] == 'brain_based' :
    simg_fgd = simg * simg_mask
  rig, intermodality_similarity = reg_opt( simg_fgd, fmri_template, transform_list=[ 'Rigid' ], 
    registration_strategy = registration_strategy[1], intensity_transform=registration_strategy[2] )
  if verbose:
    print("rsf template mask and labels")
  bmask = ants.apply_transforms( fmri_template, simg_mask, rig['invtransforms'], interpolator='genericLabel', which_to_invert=[True,False] )
  rsflabels = ants.apply_transforms( fmri_template, simg_labels, rig['invtransforms'], interpolator='genericLabel', which_to_invert=[True,False] )

  def temporal_derivative_same_shape(array):
    """
    Compute the temporal derivative of a 2D numpy array along the 0th axis (time)
    and ensure the output has the same shape as the input.

    :param array: 2D numpy array with time as the 0th axis.
    :return: 2D numpy array of the temporal derivative with the same shape as input.
    """
    derivative = np.diff(array, axis=0)
    
    # Append a row to maintain the same shape
    # You can choose to append a row of zeros or the last row of the derivative
    # Here, a row of zeros is appended
    zeros_row = np.zeros((1, array.shape[1]))
    return np.vstack((zeros_row, derivative ))

  def compute_tSTD(M, quantile, x=0, axis=0):
    stdM = np.std(M, axis=axis)
    # set bad values to x
    stdM[stdM == 0] = x
    stdM[np.isnan(stdM)] = x
    tt = round(quantile * 100)
    threshold_std = np.percentile(stdM, tt)
    return {'tSTD': stdM, 'threshold_std': threshold_std}

  def get_compcor_matrix(boldImage, mask, quantile):
    """
    Compute the compcor matrix.

    :param boldImage: The bold image.
    :param mask: The mask to apply, if None, it will be computed.
    :param quantile: Quantile for computing threshold in tSTD.
    :return: The compor matrix.
    """
    if mask is None:
        temp = ants.slice_image(boldImage, axis=boldImage.dimension - 1, idx=0)
        mask = ants.get_mask(temp)

    imagematrix = ants.timeseries_to_matrix(boldImage, mask)
    temp = compute_tSTD(imagematrix, quantile, 0)
    tsnrmask = ants.make_image(mask, temp['tSTD'])
    tsnrmask = ants.threshold_image(tsnrmask, temp['threshold_std'], temp['tSTD'].max())
    M = ants.timeseries_to_matrix(boldImage, tsnrmask)
    return M


  from sklearn.decomposition import FastICA
  def find_indices(lst, value):
    return [index for index, element in enumerate(lst) if element > value]

  def mean_of_list(lst):
    if not lst:  # Check if the list is not empty
        return 0  # Return 0 or appropriate value for an empty list
    return sum(lst) / len(lst)
  fmrispc = list( ants.get_spacing( fmri ) )
  if spa is None:
    spa = mean_of_list( fmrispc[0:3] ) * 1.0
  if spt is None:
    spt = fmrispc[3] * 0.5
      
  import numpy as np
  import pandas as pd
  import re
  import math
  # point data resources
  A = np.zeros((1,1))
  fmri = ants.iMath( fmri, 'Normalize' )
  if verbose:
      print("Begin rsfmri motion correction")
  # mot-co
  corrmo = antspymm.timeseries_reg(
    fmri, fmri_template,
    type_of_transform=type_of_motion_transform,
    total_sigma=0.5,
    fdOffset=2.0,
    trim = 8,
    output_directory=None,
    verbose=verbose,
    syn_metric='cc',
    syn_sampling=2,
    reg_iterations=[40,20,5],
    return_numpy_motion_parameters=True )
  
  if verbose:
      print("End rsfmri motion correction")
      print("=== next despiking and correlations ===")

  despiking_count = np.zeros( corrmo['motion_corrected'].shape[3] )
  if despike > 0.0:
      corrmo['motion_corrected'], despiking_count = antspymm.despike_time_series_afni( corrmo['motion_corrected'], c1=despike )

  despiking_count_summary = despiking_count.sum() / np.prod( corrmo['motion_corrected'].shape )
  high_motion_count=(corrmo['FD'] > FD_threshold ).sum()
  high_motion_pct=high_motion_count / fmri.shape[3]

  # filter mask based on TSNR
  mytsnr = antspymm.tsnr( corrmo['motion_corrected'], bmask )

  # anatomical mapping
  und = fmri_template * bmask
  # optional smoothing
  tr = ants.get_spacing( corrmo['motion_corrected'] )[3]
  smth = ( spa, spa, spa, spt ) # this is for sigmaInPhysicalCoordinates = TRUE
  simg = ants.smooth_image( corrmo['motion_corrected'], smth, sigma_in_physical_coordinates = True )

  # collect censoring indices
  hlinds = find_indices( corrmo['FD'], FD_threshold )
  if verbose:
    print("high motion indices")
    print( hlinds )
  if outlier_threshold < 1.0 and outlier_threshold > 0.0:
    fmrimotcorr, hlinds2 = antspymm.loop_timeseries_censoring( corrmo['motion_corrected'], 
      threshold=outlier_threshold, verbose=verbose )
    hlinds.extend( hlinds2 )
    del fmrimotcorr
  hlinds = list(set(hlinds)) # make unique

  # nuisance
  globalmat = ants.timeseries_to_matrix( corrmo['motion_corrected'], bmask )
  globalsignal = np.nanmean( globalmat, axis = 1 )
  del globalmat
  nuisance = ants.compcor( corrmo['motion_corrected'],
    ncompcor=nc, quantile=0.5, mask = bmask,
    filter_type='polynomial', degree=2 )[ 'components' ]

  if motion_as_nuisance:
      if verbose:
          print("include motion as nuisance")
          print( corrmo['motion_parameters'].shape )
      deriv = temporal_derivative_same_shape( corrmo['motion_parameters']  )
      nuisance = np.c_[ nuisance, corrmo['motion_parameters'], deriv ]

  if ica_components > 0:
    if verbose:
        print("include ica components as nuisance: " + str(ica_components))
    ica = FastICA(n_components=ica_components, max_iter=10000, tol=0.001, random_state=42 )
    globalmat = ants.timeseries_to_matrix( corrmo['motion_corrected'], bmask )
    nuisance_ica = ica.fit_transform(globalmat)  # Reconstruct signals
    nuisance = np.c_[ nuisance, nuisance_ica ]
    del globalmat

  nuisance = np.c_[ nuisance, globalsignal ]

  if impute:
    simgimp = antspymm.impute_timeseries( simg, hlinds, method='linear')
  else:
    simgimp = simg

  # falff/alff stuff  def alff_image( x, mask, flo=0.01, fhi=0.1, nuisance=None ):
  myfalff=antspymm.alff_image( simgimp, bmask, flo=f[0], fhi=f[1], nuisance=nuisance  )

  # bandpass any data collected before here -- if bandpass requested
  if f[0] > 0 and f[1] < 1.0:
    if verbose:
        print( "bandpass: " + str(f[0]) + " <=> " + str( f[1] ) )
    nuisance = ants.bandpass_filter_matrix( nuisance, tr = tr, lowf=f[0], highf=f[1] ) # some would argue against this
    globalmat = ants.timeseries_to_matrix( simg, bmask )
    globalmat = ants.bandpass_filter_matrix( globalmat, tr = tr, lowf=f[0], highf=f[1] ) # some would argue against this
    simg = ants.matrix_to_timeseries( simg, globalmat, bmask )

  if verbose:
    print("now regress nuisance")


  if len( hlinds ) > 0 :
    if censor:
        nuisance = antspymm.remove_elements_from_numpy_array( nuisance, hlinds  )
        simg = antspymm.remove_volumes_from_timeseries( simg, hlinds )

  gmmat = ants.timeseries_to_matrix( simg, bmask )
  gmmat = ants.regress_components( gmmat, nuisance )
  simg = ants.matrix_to_timeseries(simg, gmmat, bmask)
  mycmat = rsfmri_to_correlation_matrix_wide( simg, rsflabels )
  perafimg = antspymm.PerAF( simgimp, bmask )

  stats_alf = ants.label_stats(myfalff['alff'],rsflabels )
  stats_alf = widen_summary_dataframe(stats_alf, description='LabelValue', value='Mean', skip_first=True, prefix='rsf.alf.' )
  stats_flf = ants.label_stats(myfalff['falff'],rsflabels )
  stats_flf = widen_summary_dataframe(stats_flf, description='LabelValue', value='Mean', skip_first=True, prefix='rsf.flf.' )
  stats_prf = ants.label_stats(perafimg,rsflabels )
  stats_prf = widen_summary_dataframe(stats_prf, description='LabelValue', value='Mean', skip_first=True, prefix='rsf.prf.' )

  # structure the output data
  outdict = {}
  outdict['fmri_template'] = fmri_template
  outdict['brainmask'] = bmask
  outdict['labels'] = rsflabels
  outdict['upsampling'] = upsample
  outdict['alff'] = myfalff['alff']
  outdict['falff'] = myfalff['falff']
  # add global mean and standard deviation for post-hoc z-scoring
  outdict['alff_mean'] = (myfalff['alff'][myfalff['alff']!=0]).mean()
  outdict['alff_sd'] = (myfalff['alff'][myfalff['alff']!=0]).std()
  outdict['falff_mean'] = (myfalff['falff'][myfalff['falff']!=0]).mean()
  outdict['falff_sd'] = (myfalff['falff'][myfalff['falff']!=0]).std()

  rsfNuisance = pd.DataFrame( nuisance )
  if remove_it:
    import shutil
    shutil.rmtree(output_directory, ignore_errors=True )
  outdict['motion_corrected'] = corrmo['motion_corrected']
  outdict['nuisance'] = rsfNuisance
  outdict['PerAF'] = perafimg
  outdict['tsnr'] = mytsnr
  outdict['dvars'] = antspymm.dvars( corrmo['motion_corrected'], bmask )
  outdict['bandpass_freq_0']=f[0]
  outdict['bandpass_freq_1']=f[1]
  outdict['censor']=int(censor)
  outdict['spatial_smoothing']=spa
  outdict['outlier_threshold']=outlier_threshold
  outdict['FD_threshold']=outlier_threshold
  outdict['high_motion_count'] = high_motion_count
  outdict['high_motion_pct'] = high_motion_pct
  outdict['despiking_count_summary'] = despiking_count_summary
  outdict['FD_max'] = corrmo['FD'].max()
  outdict['FD_mean'] = corrmo['FD'].mean()
  outdict['FD_sd'] = corrmo['FD'].std()
  outdict['bold_evr'] =  antspyt1w.patch_eigenvalue_ratio( und, 512, [16,16,16], evdepth = 0.9, mask = bmask )
  outdict['n_outliers'] = len(hlinds)
  outdict['minutes_original_data'] = ( tr * fmri.shape[3] ) / 60.0 # minutes of useful data
  outdict['minutes_censored_data'] = ( tr * simg.shape[3] ) / 60.0 # minutes of useful data
  outdict['correlation'] = mycmat
  outdict['label_mean_alff'] = stats_alf
  outdict['label_mean_falff'] = stats_flf
  outdict['label_mean_peraf'] = stats_prf
  outdict['registration_result']=rig
  outdict['intermodality_similarity'] = intermodality_similarity
  return antspymm.convert_np_in_dict( outdict )


def reg_initializer( fixed, moving, n_simulations=32, max_rotation=30,
    transform=['rigid'], compreg=None, verbose=False ):
    """
    multi-start multi-transform registration solution - based on ants.registration

    fixed: fixed image

    moving: moving image

    n_simulations : number of simulations

    max_rotation : maximum rotation angle

    transform : list of transforms to loop through

    compreg : registration results against which to compare

    verbose : boolean

    """
    if True:
        output_directory = tempfile.mkdtemp()
        output_directory_w = output_directory + "/tra_reg/"
        os.makedirs(output_directory_w,exist_ok=True)
        bestmi = math.inf
        bestvar = 0.0
        myorig = list(ants.get_origin( fixed ))
        mymax = 0;
        for k in range(len( myorig ) ):
            if abs(myorig[k]) > mymax:
                mymax = abs(myorig[k])
        maxtrans = mymax * 0.05
        if compreg is None:
            bestreg=ants.registration( fixed,moving,'Translation',
                outprefix=output_directory_w+"trans")
            initx = ants.read_transform( bestreg['fwdtransforms'][0] )
        else :
            bestreg=compreg
            initx = ants.read_transform( bestreg['fwdtransforms'][0] )
        for regtx in transform:
            with tempfile.NamedTemporaryFile(suffix='.h5') as tp:
                rRotGenerator = ants.contrib.RandomRotate3D( ( max_rotation*(-1.0), max_rotation ), reference=fixed )
                for k in range(n_simulations):
                    simtx = ants.compose_ants_transforms( [rRotGenerator.transform(), initx] )
                    ants.write_transform( simtx, tp.name )
                    if k > 0:
                        reg = ants.registration( fixed, moving, regtx,
                            initial_transform=tp.name,
                            outprefix=output_directory_w+"reg"+str(k),
                            verbose=False )
                    else:
                        reg = ants.registration( fixed, moving,
                            regtx,
                            outprefix=output_directory_w+"reg"+str(k),
                            verbose=False )
                    mymi = math.inf
                    temp = reg['warpedmovout']
                    myvar = temp.numpy().var()
                    if verbose:
                        print( str(k) + " : " + regtx  + " : _var_ " + str( myvar ) )
                    if myvar > 0 :
                        mymi = ants.image_mutual_information( fixed, temp )
                        if mymi < bestmi:
                            if verbose:
                                print( "mi @ " + str(k) + " : " + str(mymi), flush=True)
                            bestmi = mymi
                            bestreg = reg
                            bestvar = myvar
        if bestvar == 0.0 and compreg is not None:
            return compreg        
        return bestreg


def reg(fixed, moving, transform_list=['Rigid'], max_rotation=30. , n_simulations=32, simple=False ):
    """
    Perform registration using `antspymm.tra_initializer` with rigid and SyN transformations.

    Parameters:
    -----------
    fixed : ANTsImage
        The fixed image for registration.
    moving : ANTsImage
        The moving image for registration.
    transform_list: list of strings
    max_rotation : float
    n_simulations : int
    
    Returns:
    --------
    dict
        A dictionary containing the registration results from `antspymm.tra_initializer`.
    
    Notes:
    ------
    - Uses 32 simulations to initialize transformations.
    - Allows a maximum rotation of 30 degrees.
    - Applies a two-step transformation: rigid followed by symmetric normalization (SyN).
    - Prints verbose output during execution.
    """
    rifi = ants.rank_intensity(fixed)
    rimi = ants.rank_intensity(moving)
    if simple:
        myreg = ants.registration( 
            ants.iMath(fixed,'Normalize'),
            ants.iMath(moving,'Normalize'), 
            'SyNBold', syn_sampling=2, sym_metric='cc', verbose=False )
    else:
        reginit = reg_initializer( rifi, rimi,
            n_simulations=n_simulations, max_rotation=max_rotation, 
            transform=transform_list, verbose=True )
        myreg = ants.registration( rifi, rimi, 
            'SyNOnly', intial_transform=reginit['fwdtransforms'][0], 
            syn_sampling=2, sym_metric='cc', verbose=False )
    mymi = ants.image_mutual_information( rifi, myreg['warpedmovout'] )
    return myreg, mymi


def reg_opt(fixed, moving, transform_list=['Rigid'], max_rotation=30., n_simulations=32,
        registration_strategy='optimal', intensity_transform='normalize', search_registration=['SyNOnly','SyN','SyNBold'], verbose=True):
    """
    Perform registration using `antspymm.reg_initializer` with rigid and SyN transformations.

    Runs a simple registration first, then iterates over user-specified `search_registration` values,
    selecting the best result based on mutual information (MI), unless `simple=True`, in which case only the simple version is run.
    
    Parameters:
    -----------
    fixed : ANTsImage
        The fixed image for registration.
    moving : ANTsImage
        The moving image for registration.
    transform_list: list of strings
    max_rotation : float
    n_simulations : int
    registration_strategy : either 'simple', 'standard' or 'optimal'
    intensity_transform : 'rank' or 'normalize'
    search_registration : list of strings
        Types of follow-on registration to test, e.g., ['SyNOnly', 'SyNBold', 'SyN']
    verbose : bool
    
    Returns:
    --------
    dict
        A dictionary containing the best registration results based on MI, or just the simple registration if `simple=True`.
    """
    # Apply intensity transform
    if intensity_transform == 'rank':
        rifi = ants.rank_intensity(fixed)
        rimi = ants.rank_intensity(moving)
    else:
        rifi = ants.iMath(fixed, 'Normalize')
        rimi = ants.iMath(moving, 'Normalize')

    if registration_strategy == 'standard':
        return reg( rifi, rimi, transform_list=[ 'Rigid' ] )

    # Run simple registration
    simple_reg = ants.registration(rifi, rimi, 'SyNBold', syn_sampling=2, sym_metric='cc', verbose=False)
    simple_mi = ants.image_mutual_information(rifi, simple_reg['warpedmovout'])
    
    if registration_strategy == 'simple':
        return simple_reg, simple_mi
    
    # Run full registration with multiple search_registration options
    reginit = reg_initializer(rifi, rimi, n_simulations=n_simulations, max_rotation=max_rotation, 
                              transform=transform_list, verbose=verbose)
    best_reg = simple_reg
    best_mi = simple_mi
    best_search = 'SyNBold'
    
    for sr in search_registration:
        full_reg = ants.registration(rifi, rimi, sr, 
                                     initial_transform=reginit['fwdtransforms'][0], 
                                     syn_sampling=2, sym_metric='cc', verbose=False)
        full_mi = ants.image_mutual_information(rifi, full_reg['warpedmovout'])
        
        if verbose:
            print(f'Search: {sr}, MI: {full_mi}')
        
        if full_mi < best_mi:
            best_reg = full_reg
            best_mi = full_mi
            best_search = sr
    
    if verbose:
        print(f'Best registration: {best_search}, MI: {best_mi}')
    
    return best_reg, best_mi


def merge_idp_dataframes(s=None, prf=None, mypet=None, dti=None, rsf=None, output_path=None):
    """
    Generate a summary dataframe by concatenating multiple sources of ROI-based imaging metrics.
    None of the inputs are required; missing data sources will be ignored.
    
    Parameters:
    -----------
    s : dict, optional
        Dictionary containing 'label_geometry' for ROI volumes.
    prf : dict, optional
        Dictionary containing 'label_mean' for ROI perfusion and CBF.
    mypet : dict, optional
        Dictionary containing 'label_mean' for ROI mean PET values.
    dti : dict, optional
        Dictionary containing 'label_mean_fa' (FA mean) and 'label_mean_md' (MD mean).
    rsf : dict, optional
        Dictionary containing 'label_mean_alff', 'label_mean_falff', 'label_mean_peraf' (resting-state metrics),
        and 'correlation' (ROI correlations at rest).
    output_path : str, optional
        If provided, saves the resulting dataframe as a CSV file.
    
    Returns:
    --------
    pd.DataFrame
        A dataframe containing the concatenated summary of all available metrics.
    """
    data_frames = []
    import pandas as pd

    if s and 'label_geometry' in s:
        data_frames.append(s['label_geometry'])
    if prf and 'label_mean' in prf:
        data_frames.append(prf['label_mean'])
    if mypet and 'label_mean' in mypet:
        data_frames.append(mypet['label_mean'])
    if dti:
        if 'label_mean_fa' in dti:
            data_frames.append(dti['label_mean_fa'])
        if 'label_mean_md' in dti:
            data_frames.append(dti['label_mean_md'])
    if rsf:
        if 'label_mean_alff' in rsf:
            data_frames.append(rsf['label_mean_alff'])
        if 'label_mean_falff' in rsf:
            data_frames.append(rsf['label_mean_falff'])
        if 'label_mean_peraf' in rsf:
            data_frames.append(rsf['label_mean_peraf'])
        if 'correlation' in rsf:
            data_frames.append(rsf['correlation'])
    
    if not data_frames:
        raise ValueError("No valid data sources provided. At least one input must contain valid data.")
    
    summary_df = pd.concat(data_frames, axis=1)
    
    if output_path:
        summary_df.to_csv(output_path)
    
    return summary_df