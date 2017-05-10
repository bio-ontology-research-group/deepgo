# DeepGO - Predicting Gene Ontology Functions

DeepGO - is a novel method for predicting protein functions using protein sequences and protein-protein interaction (PPI) networks. It uses deep neural networks to learn sequence and PPI network features and hierarchically classifies it with GO classes.

This repository contains script which were used to build and train DeepGO model together with the scripts for evaluating the model's performance.

## Dependencies
To install python dependencies run:
pip install -r requirements.txt

## Scripts
The scripts require GeneOntology in OBO Format.
* nn_hierarchical_seq.py - This script is used to build and train the model which uses only the sequence of protein as an input.
* nn_hierarchical_network.py - This script is used to build and train the model which uses sequence and PPI network embeddings of protein as an input.
* get_data.py, get_functions.py, mapping.py scripts are used to prepare raw data.
* blast.py script is used to evaluate BLAST method's performance

The online version of tool is available at http://deepgo.bio2vec.net/
