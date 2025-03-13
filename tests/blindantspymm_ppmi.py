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
myup=2.0 # spatial resolution
print(" Load in JHU atlas and labels ")
ex_path = os.path.expanduser( "~/.antspyt1w/" )
ex_path_mm = os.path.expanduser( "~/.antspymm/" )
JHU_atlas = ants.image_read( ex_path + 'JHU-ICBM-FA-1mm.nii.gz' ) # Read in JHU atlas
JHU_labels = ants.image_read( ex_path + 'JHU-ICBM-labels-1mm.nii.gz' ) # Read in JHU labels
#### Load in data ####
print("Load in subject data ...")
t1id = ex_path_mm + "sub-01_T1w.nii.gz"
rsid = ex_path_mm + "sub-01_asl.nii.gz"
print("brain extract the T1")
simg = ants.iMath( ants.image_read( t1id ) , 'Normalize' )
if not "simg_mask" in globals():
    simg_mask = antspyt1w.brain_extraction( simg )

pfimg = ants.image_read( rsid )
template = ants.image_read( ex_path_mm + 'PPMI_template0.nii.gz' )
template_mask = ants.image_read( ex_path_mm + 'PPMI_template0_brainmask.nii.gz' )
template_labels = ants.image_read( ex_path_mm + 'PPMI_template0_dkt_parcellation.nii.gz' )

if not "s" in globals():
    s = blindantspymm.structural( simg, simg_mask, template, template_mask, template_labels, upsample=myup )
    s_labels = s['labels']
    print( s['brain_mask_overlap'] )
    print( s['label_geometry'] )
    print( 'struct:intermodality_similarity ' + str(s['intermodality_similarity'] ))
    # ants.plot( simg * simg_mask, s_labels, crop=True )

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

###################
summary_df = blindantspymm.merge_idp_dataframes( s, prf, None, None, None, "/tmp/temp.csv" )
