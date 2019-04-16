# DeepGO - Predicting Gene Ontology Functions

DeepGO is a novel method for predicting protein functions using protein sequences and protein-protein interaction (PPI) networks. It uses deep neural networks to learn sequence and PPI network features and hierarchically classifies it with GO classes. PPI network features are learned using a neuro-symbolic approach for learning knowledge graph representations by [Alshahrani, et al.][1]

This repository contains script which were used to build and train the DeepGO model together with the scripts for evaluating the model's performance.

## Dependencies
To install python dependencies run:
pip install -r requirements.txt

## Scripts
The scripts require GeneOntology in OBO Format.
* nn_hierarchical_seq.py - This script is used to build and train the model which uses only the sequence of protein as an input.
* nn_hierarchical_network.py - This script is used to build and train the model which uses sequence and PPI network embeddings of protein as an input.
* get_data.py, get_functions.py, mapping.py scripts are used to prepare raw data.
* blast.py script is used to evaluate BLAST method's performance
* evaluation.py script is used to evalutate the performance of the FFPred, GOFDR and our method.

## Running
* Download the data file from http://deepgo.bio2vec.net/data/deepgo/data.tar.gz and extract data folder
* Install [diamond](https://github.com/bbuchfink/diamond) program on your system (diamond command should be available)
* run predict_all.py script with -i <input_fasta_filename> arguments
* See the results in results.tsv file

## Data
* eval_data.tar.gz - This archive contains data used to compare GOFDR, FFPred and our method.
* http://deepgo.bio2vec.net/data/deepgo/data.tar.gz - The files required to run predict_all.py script


[1]: https://doi.org/10.1093/bioinformatics/btx275

## Citation

If you use DeepGO for your research, or incorporate our learning algorithms in your work, please cite:

Maxat Kulmanov, Mohammed Asif Khan, Robert Hoehndorf; DeepGO: Predicting protein functions from sequence and interactions using a deep ontology-aware classifier, Bioinformatics, 2017. https://doi.org/10.1093/bioinformatics/btx624
