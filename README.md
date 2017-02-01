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
BP 0.447058464429 0.477137976344 0.48654056631
MF 0.477987650802 0.50942854874 0.522336688777
CC 0.494088741559 0.525368359988 0.538207206861

MF
INFO:F measure:      0.497304 0.710130 0.460592
INFO:ROC AUC:    0.956329

CC
INFO:F measure:      0.560690 0.704727 0.538064
INFO:ROC AUC:    0.971901

BP
INFO:F measure:      0.324912 0.609236 0.287488
INFO:ROC AUC:    0.914563
