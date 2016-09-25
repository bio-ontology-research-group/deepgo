# DeepGO - Predicting Gene Ontology Functions


## List of used database files
* Uniprot-swiss - Downloaded on 05/01/2016 at 9:50
* GO OBO Format file - Downloaded on 05/01/2016 at 9:55
* GO Yeast OBO Format file - Downloaded on 05/01/2016 at 9:50

## Filtered GO classes by number of annotations
* Molecular functions - 50
* Biological processes - 700
* Cellular components - 10, 50


## Performance reports training/testing - BP, MF, CC
* 80/20 - 0.802696 68983, 0.815761 82755, 0.801406 75901
* All/All - 0.821422 346554, 0.811965 414550, 0.825238 379534
* All/Human - 0.406858 12126, 0.548222 11859, 0.560793 15593
* All/Mouse - 0.434767 9321, 0.528762 10196, 0.561310 13385
* All/Yeast - 0.317909 3091, 0.395513 3375, 0.438704 4918
* All/Ecoli - 0.556044 2291, 0.602753 2847, 0.634596 2823
* All/Pombe - , 0.426044 2851, 0.467777 4565
* All/Zebrafish - 0.466866 1546, 0.500359 1623, 0.587050 2329
* All/Fly - 0.342000 1873, 0.426006 2145, 0.497306 2476

# Second experiments with organism specific functions
* All/Fly ,0.462496 2322 ,0.524430 2524


## Performance of top level functions

## Molecular functions

GO:0005488 - binding
             precision    recall  f1-score   support

          0       0.88      0.70      0.78     24276
          1       0.89      0.96      0.92     58479

avg / total       0.89      0.89      0.88     82755

GO:0045735 - nutrient reservoir activity
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     82671
          1       0.76      0.79      0.77        84

avg / total       1.00      1.00      1.00     82755

GO:0003824 - catalytic activity
             precision    recall  f1-score   support

          0       0.82      0.84      0.83     32773
          1       0.89      0.88      0.89     49982

avg / total       0.86      0.86      0.86     82755

GO:0016209 - antioxidant activity
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     82428
          1       0.77      0.85      0.81       327

avg / total       1.00      1.00      1.00     82755

GO:0000988 - transcription factor activity, protein binding
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     82315
          1       0.45      0.61      0.52       440

avg / total       1.00      0.99      0.99     82755

GO:0005215 - transporter activity
             precision    recall  f1-score   support

          0       0.99      0.95      0.97     77167
          1       0.56      0.91      0.69      5588

avg / total       0.96      0.95      0.95     82755

GO:0005198 - structural molecule activity
             precision    recall  f1-score   support

          0       0.99      0.99      0.99     72189
          1       0.96      0.95      0.95     10566

avg / total       0.99      0.99      0.99     82755

GO:0004871 - signal transducer activity
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     81684
          1       0.71      0.73      0.72      1071

avg / total       0.99      0.99      0.99     82755

GO:0060089 - molecular transducer activity
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     81882
          1       0.76      0.76      0.76       873

avg / total       0.99      0.99      0.99     82755

GO:0001071 - nucleic acid binding transcription factor activity
             precision    recall  f1-score   support

          0       0.99      0.99      0.99     80655
          1       0.64      0.74      0.69      2100

avg / total       0.98      0.98      0.98     82755

GO:0045182 - translation regulator activity
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     82740
          1       1.00      0.33      0.50        15

avg / total       1.00      1.00      1.00     82755

GO:0098772 - molecular function regulator
             precision    recall  f1-score   support

          0       1.00      0.98      0.99     81331
          1       0.37      0.76      0.50      1424

avg / total       0.99      0.97      0.98     82755

GO:0009055 - electron carrier activity
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     81073
          1       0.95      0.89      0.92      1682

avg / total       1.00      1.00      1.00     82755


## Cellular Component

GO:0099512 - supramolecular fiber
             precision    recall  f1-score   support

          0       1.00      0.99      0.99     75351
          1       0.38      0.69      0.49       550

avg / total       0.99      0.99      0.99     75901

GO:0031974 - membrane-enclosed lumen
             precision    recall  f1-score   support

          0       1.00      0.99      0.99     75164
          1       0.44      0.57      0.49       737

avg / total       0.99      0.99      0.99     75901

GO:0031012 - extracellular matrix
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     75645
          1       0.60      0.68      0.64       256

avg / total       1.00      1.00      1.00     75901

GO:0045202 - synapse
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     75726
          1       0.66      0.59      0.62       175

avg / total       1.00      1.00      1.00     75901

GO:0009295 - nucleoid
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     75519
          1       0.83      0.86      0.84       382

avg / total       1.00      1.00      1.00     75901

GO:0030054 - cell junction
             precision    recall  f1-score   support

          0       1.00      0.99      0.99     75054
          1       0.40      0.62      0.48       847

avg / total       0.99      0.99      0.99     75901

GO:0016020 - membrane
             precision    recall  f1-score   support

          0       0.91      0.96      0.94     57773
          1       0.86      0.71      0.78     18128

avg / total       0.90      0.90      0.90     75901

GO:0043226 - organelle
             precision    recall  f1-score   support

          0       0.91      0.91      0.91     56160
          1       0.75      0.74      0.75     19741

avg / total       0.87      0.87      0.87     75901

GO:0044456 - synapse part
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     75503
          1       0.56      0.71      0.62       398

avg / total       1.00      1.00      1.00     75901

GO:0019012 - virion
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     75589
          1       0.58      0.60      0.59       312

avg / total       1.00      1.00      1.00     75901

GO:0032991 - macromolecular complex
             precision    recall  f1-score   support

          0       0.95      0.95      0.95     56435
          1       0.86      0.86      0.86     19466

avg / total       0.93      0.93      0.93     75901

GO:0044217 - other organism part
             precision    recall  f1-score   support

          0       0.99      0.97      0.98     74349
          1       0.29      0.66      0.40      1552

avg / total       0.98      0.96      0.97     75901

GO:0044464 - cell part
             precision    recall  f1-score   support

          0       0.65      0.63      0.64      7242
          1       0.96      0.96      0.96     68659

avg / total       0.93      0.93      0.93     75901

GO:0005623 - cell
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     75736
          1       0.65      0.36      0.47       165

avg / total       1.00      1.00      1.00     75901

GO:0005576 - extracellular region
             precision    recall  f1-score   support

          0       0.98      0.99      0.99     71836
          1       0.83      0.73      0.78      4065

avg / total       0.98      0.98      0.98     75901

GO:0044425 - membrane part
             precision    recall  f1-score   support

          0       0.95      0.98      0.97     58382
          1       0.94      0.83      0.88     17519

avg / total       0.95      0.95      0.95     75901

GO:0044420 - extracellular matrix component
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     75875
          1       0.52      0.50      0.51        26

avg / total       1.00      1.00      1.00     75901

GO:0044421 - extracellular region part
             precision    recall  f1-score   support

          0       0.98      0.98      0.98     73468
          1       0.43      0.48      0.45      2433

avg / total       0.96      0.96      0.96     75901

GO:0044422 - organelle part
             precision    recall  f1-score   support

          0       0.91      0.96      0.93     61794
          1       0.77      0.58      0.66     14107

avg / total       0.88      0.89      0.88     75901

GO:0044423 - virion part
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     75058
          1       0.76      0.72      0.74       843

avg / total       0.99      0.99      0.99     75901

Protein centric F measure:      0.801406 75901


GO:0007610 - behavior
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     68971
          1       0.33      0.08      0.13        12

avg / total       1.00      1.00      1.00     68983

GO:0023052 - signaling
             precision    recall  f1-score   support

          0       1.00      0.99      1.00     68742
          1       0.24      0.46      0.32       241

avg / total       1.00      0.99      0.99     68983

GO:0071840 - cellular component organization or biogenesis
             precision    recall  f1-score   support

          0       0.98      0.97      0.98     63543
          1       0.71      0.77      0.74      5440

avg / total       0.96      0.96      0.96     68983

GO:0048511 - rhythmic process
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     68936
          1       0.76      0.62      0.68        47

avg / total       1.00      1.00      1.00     68983

GO:0065007 - biological regulation
             precision    recall  f1-score   support

          0       0.92      0.96      0.94     58473
          1       0.72      0.56      0.63     10510

avg / total       0.89      0.90      0.90     68983

GO:0044699 - single-organism process
             precision    recall  f1-score   support

          0       0.90      0.83      0.86     35833
          1       0.83      0.90      0.86     33150

avg / total       0.87      0.86      0.86     68983

GO:0001906 - cell killing
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     68851
          1       0.57      0.59      0.58       132

avg / total       1.00      1.00      1.00     68983

GO:0022610 - biological adhesion
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     68447
          1       0.62      0.66      0.64       536

avg / total       0.99      0.99      0.99     68983

GO:0032502 - developmental process
             precision    recall  f1-score   support

          0       0.99      0.99      0.99     67282
          1       0.54      0.53      0.54      1701

avg / total       0.98      0.98      0.98     68983

GO:0032501 - multicellular organismal process
             precision    recall  f1-score   support

          0       0.99      1.00      1.00     68352
          1       0.57      0.43      0.49       631

avg / total       0.99      0.99      0.99     68983

GO:0051704 - multi-organism process
             precision    recall  f1-score   support

          0       0.99      0.98      0.99     66692
          1       0.55      0.68      0.61      2291

avg / total       0.97      0.97      0.97     68983

GO:0009987 - cellular process
             precision    recall  f1-score   support

          0       0.76      0.50      0.60     11981
          1       0.90      0.97      0.93     57002

avg / total       0.88      0.89      0.88     68983

GO:0022414 - reproductive process
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     68469
          1       0.69      0.70      0.69       514

avg / total       1.00      1.00      1.00     68983

GO:0008152 - metabolic process
             precision    recall  f1-score   support

          0       0.83      0.56      0.67     15627
          1       0.88      0.97      0.92     53356

avg / total       0.87      0.87      0.86     68983

GO:0051179 - localization
             precision    recall  f1-score   support

          0       0.97      0.99      0.98     62558
          1       0.84      0.75      0.79      6425

avg / total       0.96      0.96      0.96     68983

GO:0040007 - growth
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     68828
          1       0.36      0.26      0.30       155

avg / total       1.00      1.00      1.00     68983

GO:0040011 - locomotion
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     68240
          1       0.86      0.66      0.75       743

avg / total       0.99      1.00      0.99     68983

GO:0050896 - response to stimulus
             precision    recall  f1-score   support

          0       0.97      0.96      0.97     62479
          1       0.66      0.74      0.70      6504

avg / total       0.94      0.94      0.94     68983

GO:0002376 - immune system process
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     68325
          1       0.60      0.57      0.58       658

avg / total       0.99      0.99      0.99     68983
