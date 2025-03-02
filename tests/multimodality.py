######################
import ants          #
import antspymm      #
import curvanato     #
import re            #
import random        #
import blindantspymm #
random.seed(42)  # Any fixed integer seed
ddir = tdir = "/Users/stnava/Downloads/Ferret/"
num='053'
simg = ants.image_read( ddir + 'NiftiMRI/sub-PTCRA'+num+'/anat/sub-PTCRA'+num+'_run-001_t1.nii.gz' )
simg_mask = ants.image_read( ddir + 'ProcessedMRI/sub-PTCRA'+num+'/anat/sub-PTCRA'+num+'_run-001_t1mask.nii.gz' )
rimg = ants.image_read( ddir + 'NiftiMRI/sub-PTCRA'+num+'/func/sub-PTCRA'+num+'_run-001_rsfmri.nii.gz' )
pfimg = ants.image_read( ddir + 'NiftiMRI/sub-PTCRA'+num+'/perf/sub-PTCRA'+num+'_run-001_perf.nii.gz' )

dwifn = ddir + 'NiftiMRI/sub-PTCRA'+num+'/dwi/sub-PTCRA'+num+'_run-001_dti.nii.gz'
dwibval = re.sub( 'nii.gz' , 'bval' , dwifn )
dwibvec = re.sub( 'nii.gz' , 'bvec' , dwifn )
dimg = ants.image_read( dwifn )
template = ants.image_read( tdir + 'TemplateT1/S_template0.nii.gz' )
template_mask = ants.image_read( tdir + 'TemplateT1/S_template0_mask.nii.gz' )
ptem = ants.image_read( tdir + './FerretTemplate/evDTI/evDTI_template_FA.nii' )
# ptem = ants.image_read( tdir + './FerretTemplate/evT2/evT2_template.nii' )
ptemlab0 = ants.image_read( tdir + './FerretTemplate/evDTI/AtlasMasks/evDTI_SEGMENTATION.nii' )


if not "tlab" in globals():
    tlab=blindantspymm.template_based_labeling( template, template_mask, ptem, ptemlab0, type_of_transform='SyN' )
    timgbc=tlab['transformed_template']
    ptemlab=tlab['transformed_labels']
 #   ants.plot( timgbc, ptemlab, crop=True )
    ants.image_write( timgbc, '/tmp/tempb.nii.gz')
    ants.image_write( ptemlab, '/tmp/templab.nii.gz')
    template_labels = tlab['transformed_labels_orig_space']

# here i am using these data-driven labels rather than the prior-based labels 
# because i believe they are a little more generically relevant for the resolution 
# of the data on hand
# template_labels = curvanato.cluster_image_gradient_prop( template_mask, n_clusters=8, sigma=1.0 )
if not "s" in globals():
    s = blindantspymm.structural( simg, simg_mask, template, template_mask, template_labels )
    s_labels = s['labels']
    print( s['brain_mask_overlap'] )
    print( s['label_geometry'] )
#    ants.plot( simg * simg_mask, s_labels, crop=True )

if not "mypet" in globals():
    pet3d = ants.get_average_of_timeseries( pfimg )  # this is a placeholder
    mypet = blindantspymm.pet( pet3d, simg, simg_mask, s_labels, verbose=True )
    ants.image_write( mypet['registration_result']['warpedmovout'], '/tmp/x1pet.nii.gz' )
    ants.image_write( mypet['pet3d'], '/tmp/x0pet.nii.gz' )

zonko

if not "rsf" in globals(): # not implemented yet
    print("Begin rsf")
    rimgsub = antspymm.remove_volumes_from_timeseries( rimg, range(50,1000) )
    rsf = blindantspymm.rsfmri( rimgsub, simg, simg_mask, s_labels, verbose=True )
    ants.image_write( rsf['registration_result']['warpedmovout'], '/tmp/x1rsf.nii.gz' )
    ants.image_write( rsf['fmri_template'], '/tmp/x0rsf.nii.gz' )
    print( rsf['correlation'] )


if not "prf" in globals():
    prf = blindantspymm.perfusion( pfimg, simg, simg_mask, s_labels, nc=4,
        tc='alternating', verbose=True )
    print(prf.keys())
    ants.image_write( prf['registration_result']['warpedmovout'], '/tmp/x1prf.nii.gz' )
    ants.image_write( prf['meanBold'], '/tmp/x0prf.nii.gz' )

if not "dti" in globals():
    dti = blindantspymm.dwi( dimg, simg, simg_mask, s_labels, dwibval, dwibvec )
    ants.image_write( dti['registration_result']['warpedmovout'], '/tmp/x1dti.nii.gz' )
    ants.image_write( dti['dwimean'], '/tmp/x0dti.nii.gz' )

import pandas as pd
summary_df = pd.concat( [
   s['label_geometry'],     # ROI volumes
   prf['label_mean'],       # ROI perfusion and cbf
   mypet['label_mean'],     # ROI mean PET values
   dti['label_mean_fa'],    # ROI FA mean
   dti['label_mean_md'],    # ROI MD mean
   rsf['label_mean_alff'],  # alff, falff and peraf
   rsf['label_mean_falff'], #
   rsf['label_mean_peraf'], ##################################
   rsf['correlation'] ], axis=1 ) # ROI correlations at rest #
##############################################################
summary_df.to_csv( "/tmp/temp.csv" )
##############################################################
