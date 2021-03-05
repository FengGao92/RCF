This folder contains one licence, one jupyter notebook, one csv data file, two .npy files, and three scripts that are necessary to reproduce results from paper "Direct prediction of bioaccumulation of organic contaminants in plant roots from soils with machine learning models based on molecular structures". 
=====================================================================================================
Specifically, this folder contains:

1. RCF_soil_prediction_model.py: this script can be used for comparing three models in the paper: LR, GBRT-property, and GBRT-ECFP.
2. RCF_soil_prediction_data_analysis.py: this script can be used for basic statistics description of dataset, computation of dice coeffcient and clustering of molecules based on ECFP.
3. RCF_soil_prediction_visualize_mol_substructure.ipynb: this jupyter notebook can be used to visualize important substructures of chemicals
4. Utility.py: this script includes several utility functions required by other scripts
5. sample_index.npy: this file contains a sample index of data points used by other scripts
6. chemical_group.npy: this file contains chemical labels from clustering used by other scripts
7. RCF_soil.csv: this csv data file contains basic 
8. LICENCE: GNU GENERAL PUBLIC LICENSE

=====================================================================================================
System Requirements

Windows (10), Mac OS X (>= 10.8) or Linux
Python >= 3.7

=====================================================================================================
Dependencies

To run the codes in this folder requires installation of Python RDKit 2020.3.3, Scikit-learn 0.24.1, Seaborn 0.10.1, Pandas 1.1.1 packages and Jupyter Notebook.

=====================================================================================================
Installation of Python packages

We suggest users to create a virtual environment to install the following packages.

RDKit can be installed with:
	conda: conda install -c conda-forge rdkit
	More information about installation can be found at: https://www.rdkit.org/docs/Install.html
	Installation should be within 10 minutes 
Scikit-learn can be installed with:
	conda: conda install -c conda-forge scikit-learn
	pip: pip install -U scikit-learn
	More information about installation can be found at: https://scikit-learn.org/stable/install.html
	Installation should be within 10 minutes
Seaborn can be installed with:
	pip: pip install seaborn==0.10.1
	More information about installation can be found at: https://seaborn.pydata.org/installing.html
	Installation should be within 10 minutes
Pandas can be installed with:
	conda: conda install pandas
	pip: pip install pandas
	More information about installation can be found at: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
	Installation should be within 10 minutes
Jupyter Notebook can be installed with:
	codna: conda install -c conda-forge notebook
	pip: pip install notebook
	More information about installation can be found at: https://jupyter.org/install
	Installation should be within 20 minutes
=====================================================================================================
Quick start

1. To train and test three models in the paper, in terminal, run: python RCF_soil_prediction_model.py
2. To produce heatmap of chemical similarity comparion, statistics description of dataset and cluster chemicals, in termial, run: python RCF_soil_prediction_data_analysis.py
3. To visualize substructure in molecules, in terminal, run: jupyter notebook. Then load RCF_soil_prediction_visualize_mol_substructure.ipynb.
The running time for each of the script should be fewer than 30 minutes, depending on the hardwares.
*The default for all figure outputs is disabled. To output and save figures in the current folder, please follow instructions in the scripts to comment out certain lines.
=====================================================================================================
License of use

GNU GENERAL PUBLIC LICENSE

