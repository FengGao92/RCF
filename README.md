# RCF

Todo: 1. update codes; 2. add more description; 3. add figues


=======================================================================================
System Requirements

Windows (10), Mac OS X (>= 10.8) or Linux
Python >= 3.7

=======================================================================================
Dependencies

To run the codes in this folder requires installation of Python RDKit 2020.3.3, Scikit-learn 0.24.1, Seaborn 0.10.1, Pandas 1.1.1 packages and Jupyter Notebook.

=======================================================================================
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
========================================================================================
Quick start

1. To train and test models in the paper, in terminal, run each model's script.
2. To produce heatmap of chemical similarity comparion, statistics description of dataset and cluster chemicals, in termial, run: python RCF_soil_prediction_data_analysis.py
3. To visualize substructure in molecules, in terminal, run: jupyter notebook. Then load RCF_soil_prediction_visualize_mol_substructure.ipynb.
4. Run models repeatedly (10 times) to collect results.
5. Uncomment to output figures.
=========================================================================================
Introduction

=========================================================================================
Results
