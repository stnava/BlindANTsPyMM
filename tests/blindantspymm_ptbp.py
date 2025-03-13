######################
import ants          #
import antspymm      #
import curvanato     #
import re            #
import random        #
import antspyt1w     #
import antspymm      #
import blindantspymm #
import pandas as pd  #
import os            #
random.seed(42)  # Any fixed integer seed
# data https://figshare.com/articles/dataset/PTBP_Nifti_1/1044196
myup=2.0 # spatial resolution
print(" Load in JHU atlas and labels ")
ex_path = os.path.expanduser( "~/.antspyt1w/" )
ex_path_mm = os.path.expanduser( "~/.antspymm/" )
data_path = "./data/PEDS010/"
JHU_atlas = ants.image_read( ex_path + 'JHU-ICBM-FA-1mm.nii.gz' ) # Read in JHU atlas
JHU_labels = ants.image_read( ex_path + 'JHU-ICBM-labels-1mm.nii.gz' ) # Read in JHU labels
#### Load in data ####
print("Load in subject data ...")
t1id = data_path + "20101008/Anatomy/PEDS010_20101008_mprage_t1.nii.gz"
rsid = data_path + "20100722/BOLD/PEDS010_20100722_bold_fc_1.nii.gz"
asid = data_path + "20101008/PCASL/PEDS010_20101008_pcasl_1.nii.gz"
dwid = data_path + "20101008/DWI/PEDS010_20101008_0014_DTI_1_1x0_30x1000.nii.gz"
dwibval = re.sub( 'nii.gz' , 'bval' , dwid )
dwibvec = re.sub( 'nii.gz' , 'bvec' , dwid )
print("brain extract the T1")
simg = ants.iMath( ants.image_read( t1id ) , 'Normalize' )
rsimg = ants.image_read( rsid )
pfimg = ants.image_read( asid )
dwimg = ants.image_read( dwid )
if not "simg_mask" in globals():
    simg_mask = antspyt1w.brain_extraction( simg )

template = ants.image_read( ex_path_mm + 'PPMI_template0.nii.gz' )
template_mask = ants.image_read( ex_path_mm + 'PPMI_template0_brainmask.nii.gz' )
template_labels = ants.image_read( ex_path_mm + 'PPMI_template0_dkt_parcellation.nii.gz' )

if not "s" in globals():
    s = blindantspymm.structural( simg, simg_mask, template, template_mask, template_labels, upsample=myup )
    s_labels = s['labels']
    print( s['brain_mask_overlap'] )
    print( s['label_geometry'] )
    print( 'struct:intermodality_similarity ' + str(s['intermodality_similarity'] ))
    ants.plot( simg * simg_mask, s_labels, crop=True )


if not "prf" in globals():
    # we use a different registration strategy here vs in the non-human example
    prf = blindantspymm.perfusion( pfimg, simg, simg_mask, s_labels, nc=8,
        tc='alternating', verbose=True, upsample=myup, 
        registration_strategy=['brain_based', 'optimal', 'normalize'] )
    print(prf.keys())
    ants.image_write( simg, '/tmp/x1str.nii.gz' )
    ants.image_write( prf['registration_result']['warpedmovout'], '/tmp/x1prf.nii.gz' )
    ants.plot( simg * simg_mask, prf['registration_result']['warpedmovout'], crop=True )
    ants.image_write( prf['meanBold'], '/tmp/x0prf.nii.gz' )
    print( 'prf:intermodality_similarity ' + str(prf['intermodality_similarity'] ))
    # mydf=pd.DataFrame( ants.timeseries_to_matrix(pfimg, prf['brainmask'] ) )
    # mydf.to_csv("/tmp/mat2.csv")
    # prf['nuisance'].to_csv("/tmp/nuis.csv")
    ants.image_write( prf['cbf'], '/tmp/tempcbf.nii.gz' )


if not "rsf" in globals():
    print("Begin rsf")
    rimgsub = antspymm.remove_volumes_from_timeseries( rsimg, range(100,1000) )
    rsf = blindantspymm.rsfmri( rimgsub, simg, simg_mask, s_labels, upsample=myup, 
        registration_strategy=['brain_based', 'optimal', 'normalize'],
        verbose=True )
    ants.image_write( rsf['registration_result']['warpedmovout'], '/tmp/x1rsf.nii.gz' )
    ants.image_write( rsf['fmri_template'], '/tmp/x0rsf.nii.gz' )
    print( rsf['correlation'] )
    print( 'rsf:intermodality_similarity ' + str(rsf['intermodality_similarity'] ))


if not "dti" in globals():
    dti = blindantspymm.dwi( dwimg, simg, simg_mask, s_labels, dwibval, dwibvec, upsample=myup, 
        registration_strategy=['head_based', 'optimal', 'normalize'], )
    ants.image_write( dti['registration_result']['warpedmovout'], '/tmp/x1dti.nii.gz' )
    ants.image_write( dti['dwimean'], '/tmp/x0dti.nii.gz' )
    print( 'dti:intermodality_similarity ' + str(dti['intermodality_similarity'] ))

###################
mypet=None
summary_df = blindantspymm.merge_idp_dataframes( s, prf, mypet, dti, rsf, "/tmp/temp.csv" )

