######################
import blindantspymm #
import ants          #
import antspymm      #
import curvanato     #
import re            #
ddir = "/Users/stnava/Downloads/Ferret/"
simg = ants.image_read( ddir + 'NiftiMRI/sub-PTCRA010/anat/sub-PTCRA010_run-001_t1.nii.gz' )
simg_mask = ants.image_read( ddir + 'mask.nii.gz' )
rimg = ants.image_read( ddir + 'NiftiMRI/sub-PTCRA010/func/sub-PTCRA010_run-001_rsfmri.nii.gz' )
dwifn = ddir + 'NiftiMRI/sub-PTCRA010/dwi/sub-PTCRA010_run-001_dti.nii.gz'
dwibval = re.sub( 'nii.gz' , 'bval' , dwifn )
dwibvec = re.sub( 'nii.gz' , 'bvec' , dwifn )
dimg = ants.image_read( dwifn )
template = ants.image_read( ddir + 'TemplateT1/S_template0.nii.gz' )
template_mask = ants.image_read( ddir + 'TemplateT1/S_template0_mask.nii.gz' )
timgb = template*template_mask
ptem = ants.image_read( ddir + './FerretTemplate/evDTI/evDTI_template_FA.nii' )
ptem = ants.image_read( ddir + './FerretTemplate/evT2/evT2_template.nii' )
ptemreg = ants.registration( timgb, ptem, 'SyN' )
ptemlab = ants.image_read( ddir + './FerretTemplate/evDTI/AtlasMasks/evDTI_SEGMENTATION.nii' )
ptemlab = ants.apply_transforms( timgb,  ptemlab, ptemreg['fwdtransforms'],
    interpolator='nearestNeighbor' )
template_labels = curvanato.cluster_image_gradient_prop( template_mask, n_clusters=8, sigma=1.0 )
s = blindantspymm.structural( simg, simg_mask, template, template_mask, template_labels )
print( s['brain_mask_overlap'] )
ants.plot( simg * simg_mask, s['inverse_warped_labels'], crop=True )


# now dwi
# dipy_dti_recon(image, bvalsfn, bvecsfn, mask=None, b0_idx=None, mask_dilation=2, mask_closing=5, fit_method='WLS', trim_the_mask=2.0, free_water=False, verbose=False)
dwimean = ants.get_average_of_timeseries( dimg )
dwimeanK = ants.threshold_image( dwimean, 'Otsu', 2 )
simgK = ants.threshold_image( simg, 'Otsu', 2 )
# dwireg = ants.registration( simg, dwimean, 'BOLDRigid') 
dwireg = antspymm.tra_initializer( simg, dwimean, n_simulations=32, 
    max_rotation=30, transform=['rigid'], verbose=True )

ants.image_write(dwireg['warpedfixout'], '/tmp/temp2.nii.gz')
dwimask = ants.apply_transforms( dwimean, simg_mask, dwireg['invtransforms'], 
    whichtoinvert=[True],
    interpolator='nearestNeighbor' )
ants.plot( dwimean* dwimask, crop=True )
dti = antspymm.dipy_dti_recon( dimg, dwibval, dwibvec, mask=dwimask )

# now rsfmri
rsfmean = ants.get_average_of_timeseries( rimg )
rsfreg = antspymm.tra_initializer( simg, rsfmean, n_simulations=32, 
    max_rotation=30, transform=['rigid'], verbose=True )
# ants.image_write(rsfmean, '/tmp/temp0.nii.gz')
# ants.image_write(rsfreg['warpedfixout'], '/tmp/temp2.nii.gz')
rsfmask = ants.apply_transforms( rsfmean, simg_mask, rsfreg['invtransforms'], 
    whichtoinvert=[True],
    interpolator='nearestNeighbor' )

