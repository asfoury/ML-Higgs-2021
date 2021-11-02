# ML-Higgs-2021

### Team 
Ahmed Ezzo : ahmed.ezzo@epfl.ch
Emna Fendri : emna.fendri@epfl.ch
Mohamed Elasfoury : mohamed.elasfoury@epfl.ch

### Data
The data is in the data directory and should be extracted before any code is executed.

## Project Structure

All our code is in the scripts directory.

### Notebooks

Simulation_ridge.ipynb : Notebook to run (Processing+training) for ridge regression (that gave us the best accuracy result).

cross_validation_raw.ipynb : This notebook contains the cross validation to find lambda for reg logistic regression using the raw data

crossforRidge.ipynb : Notebook showing cross validation to find the best parameter lambda for ridge regression for each partition according to the accuracy. We also test for several values of degree for the polynomial expansion.

data_exploration.ipynb : This notebook shows the code that we used to get more insight over the given data. It Also showcases the code we used to generate some graphs (Number of missing values by Jet number, accuracies of multiple feature expansion techniques).

project1.ipynb : This notebook contains the first accuracies we obtained using the 6 ML methods without processing the data.

### Python Files

run.py : code that outputs our best prediction

implementations.py : our implementation of the 6 ML methods.

plots.py : cross validation plot

proj1_helpers.py : helper functions (create output csv, calculate accuracy of a model, load the data and others)  

cross_validation_ridge_raw.py : cross validation using ridge regression without processing data

costs.py : Helper script that contains the cost functions

process.py : This file contains the function `process` that we use to clean the data processing. All the data processing techniques described by the paper are implemented in this file. 

process_data_exploration.py : Modified process.py file. Made for the data_exploration.ipynb notebook so we can create graphs more easily.




