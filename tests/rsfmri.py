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
num='040'
simg = ants.image_read( ddir + 'NiftiMRI/sub-PTCRA'+num+'/anat/sub-PTCRA'+num+'_run-001_t1.nii.gz' )
sbody = ants.threshold_image( simg, 'Otsu', 1 )
simg_mask = ants.image_read( ddir + 'ProcessedMRI/sub-PTCRA'+num+'/anat/sub-PTCRA'+num+'_run-001_t1mask.nii.gz' )
rimg = ants.image_read( ddir + 'NiftiMRI/sub-PTCRA'+num+'/func/sub-PTCRA'+num+'_run-001_rsfmri.nii.gz' )
pfimg = ants.image_read( ddir + 'NiftiMRI/sub-PTCRA'+num+'/perf/sub-PTCRA'+num+'_run-001_perf.nii.gz' )

template = ants.image_read( tdir + 'TemplateT1/S_template0.nii.gz' )
template_mask = ants.image_read( tdir + 'TemplateT1/S_template0_mask.nii.gz' )
# template_mask = ants.threshold_image( template, 'Otsu', 1 )

if not "template_labels" in globals():
    template_labels = curvanato.cluster_image_gradient_prop( template_mask, n_clusters=8, sigma=1.0 )
#    ants.plot( template, template_labels )

if not "s" in globals():
    s = blindantspymm.structural( simg, simg_mask, template, template_mask, template_labels )
    s_labels = s['labels']
    print( s['brain_mask_overlap'] )
    print( s['label_geometry'] )


if not "prf" in globals():
    prf = blindantspymm.perfusion( pfimg, simg, simg_mask, s_labels, nc=4,
        tc='alternating', verbose=True )
    print(prf.keys())
    ants.image_write( prf['registration_result']['warpedmovout'], '/tmp/x1prf.nii.gz' )
    ants.image_write( prf['meanBold'], '/tmp/x0prf.nii.gz' )


if not "rsf" in globals(): # not implemented yet
    print("Begin rsf")
    rimgsub = antspymm.remove_volumes_from_timeseries( rimg, range(50,1000) )
    rsf = blindantspymm.rsfmri( rimgsub, simg, simg_mask, s_labels, verbose=True )
    ants.image_write( rsf['registration_result']['warpedmovout'], '/tmp/x1rsf.nii.gz' )
    ants.image_write( rsf['fmri_template'], '/tmp/x0rsf.nii.gz' )
    print( rsf['correlation'] )

