######################
import blindantspymm #
import ants          #
import antspymm      #
import curvanato     #
import re            #
import random        #
random.seed(42)  # Any fixed integer seed
tdir = "/Users/stnava/Downloads/Ferret/"
ddir = "/Users/stnava/Downloads/Ferret2/"
num='012'
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

# here i am using these data-driven labels rather than the prior-based labels 
# because i believe they are a little more generically relevant for the resolution 
# of the data on hand
template_labels = curvanato.cluster_image_gradient_prop( template_mask, n_clusters=8, sigma=1.0 )
if not "s" in globals():
    s = blindantspymm.structural( simg, simg_mask, template, template_mask, template_labels )
    s_labels = s['inverse_warped_labels']
    print( s['brain_mask_overlap'] )
    print( s['label_geometry'] )
#    ants.plot( simg * simg_mask, s_labels, crop=True )

if not "rsf" in globals(): # not implemented yet
    print("Begin rsf")
    rimgsub = antspymm.remove_volumes_from_timeseries( rimg, range(55,1000) )
    rsf = blindantspymm.rsfmri( rimgsub, simg, simg_mask, s_labels, verbose=True )
    print( rsf['label_geometry'] )
    
if not "mypet" in globals():
    pet3d = ants.get_average_of_timeseries( pfimg )  # this is a placeholder
    mypet = blindantspymm.pet( pet3d, simg, simg_mask, s_labels, verbose=True )

if not "prf" in globals():
    prf = blindantspymm.perfusion( pfimg, simg, simg_mask, s_labels, verbose=True )
    print(prf.keys())

if not "dti" in globals():
    dti = blindantspymm.dwi( dimg, simg, simg_mask, s_labels, dwibval, dwibvec )


