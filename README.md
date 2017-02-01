# DeepGO - Predicting Gene Ontology Functions


## List of used database files
* Uniprot-swiss - Downloaded on 05/01/2016 at 9:50
* GO OBO Format file - Downloaded on 05/01/2016 at 9:55
* GO Yeast OBO Format file - Downloaded on 05/01/2016 at 9:50

## Filtered GO classes by number of annotations
* Molecular functions - 50
* Biological processes - 250
* Cellular components - 50

## Blast baseline
ALL 0.447146100525 0.505558369319 0.527824422518
BP 0.368905461174 0.393726591066 0.401485457218
MF 0.353407015961 0.376653294188 0.386197112411
CC 0.404517201424 0.430126252277 0.440637591576

MF
INFO:F measure:      0.497304 0.710130 0.460592
INFO:ROC AUC:    0.956329

CC
INFO:F measure:      0.560690 0.704727 0.538064
INFO:ROC AUC:    0.971901

BP
INFO:F measure:      0.324912 0.609236 0.287488
INFO:ROC AUC:    0.914563
