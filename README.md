# Self-supervised molecular pretraining strategy for low-resource reaction prediction scenarios
This is the code for "Self-supervised molecular pretraining strategy for low-resource reaction prediction scenarios" paper. The preprint of this paper can be found in ChemRxiv with DOI: 10.33774/chemrxiv-2021-fxvwg-v2

# Conda Environemt Setup
conda env create -f environment.yml


# Dataset
The data for training, dev and testing of Baeyer-Villiger reaction are provided in ```data/Baeyer-Villiger reaction``` file. 
The data for training, dev and testing of Heck prediction are provided in ```data/Heck reaction``` file.
The data for training, dev and testing of Sharpless prediction are provided in ```data/ Sharpless reaction``` file.
The molecular ChEMBL data can be found https://www.ebi.ac.uk/chembl/
The molecular ZINC data can be found http://zinc.docking.org/tranches/home/

# Quickstart
# Step 1: train the model in different reaction prediction
run train_transformer.py 
get a baseline model

# Step 2. train the molecule-pretrained model in different reaction prediction 
run train_downstream.py 
get the molecule-pretrained-Mass model

# Step 3. test
run test.py
