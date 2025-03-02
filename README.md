# blindANTsPyMM

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/stnava/blindANTsPyMM/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/stnava/blindANTsPyMM/tree/main)

![mapping](https://i.imgur.com/qKqYjU9.jpeg)

## processing utilities for timeseries/multichannel images - few to no assumptions about species or organ system

the outputs of these processes can be used for data inspection/cleaning/triage
as well for interrogating hypotheses.

this package also keeps track of the latest preferred algorithm variations for
production environments.

here is an example:

https://github.com/stnava/BlindANTsPyMM/blob/main/tests/multimodality.py

​it shows how one can:

* propagate structural labels between a structural image and a mismatched template (pierpaoli) 

* process structural, perfusion, resting bold and dw MRI ... the standard outputs for each of these are based on the labels that the user passes and these are summarized into a single row csv with reasonably named values. 

* a draft / placeholder for PET processing ...  if you can share some data for that, then i can add a real data example

​example output names in the output dictionaries:


```python

>>> s['label_geometry']
   smri.Label1.VolumeInMillimeters  ...  smri.Label8.SurfaceAreaInMillimetersSquared
0                      1494.628929  ...                                   402.816543

>>> rsf['label_mean_falff']
   rsf.flf.LabelValue1.Mean  rsf.flf.LabelValue2.Mean  ...  rsf.flf.LabelValue7.Mean  rsf.flf.LabelValue8.Mean
0                  0.988382                  1.157792  ...                  0.933799                  0.796773

>>> rsf['correlation']
   Corr_Label1_Label2  Corr_Label1_Label3  ...  Corr_Label6_Label8  Corr_Label7_Label8
0            0.672981            -0.02548  ...           -0.579755           -0.259678


>>> dti['label_mean_fa']
   dwi.fa.LabelValue0.Mean  dwi.fa.LabelValue1.Mean  ...  dwi.fa.LabelValue7.Mean  dwi.fa.LabelValue8.Mean
0                 0.000016                 0.143126  ...                 0.171282                 0.190617

>>> mypet['label_mean']
   pet.LabelValue1.Mean  pet.LabelValue2.Mean  ...  pet.LabelValue7.Mean  pet.LabelValue8.Mean
0            285.087423            339.381761  ...             187.33307            201.550265


```


things to write out 

```python
import pandas as pd
summary_df = pd.concat( [
   s['label_geometry'],
   prf['label_mean'],
   mypet['label_mean'],
   dti['label_mean_fa'],
   dti['label_mean_md'],
   rsf['label_mean_alff'],
   rsf['label_mean_falff'],
   rsf['label_mean_peraf'],
   rsf['correlation'] ], axis=1 )
```

eventually will provide `write_qc` and `write_roi_summary` functions 
that formalize the output style.
