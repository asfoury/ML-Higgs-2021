# Useful starting lines

import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *
from implementations import *
from costs import *


def filter_on_jet(y,x,jet_num):
    mask = x[:,22]== jet_num
    x_jet= x[mask]
    y_jet = y[mask]
    return x_jet,y_jet

def clean(tX):
    mass_features = [0,1,2,5]
    for feature in range(tX.shape[1]):
        if(feature == 29):
            col = tX[:, feature]
            clean = col[col != 0]
            avg = np.mean(clean)
            #print(col[col == 0].shape , feature )
            col[col == 0] = avg
        else:
            col = tX[:, feature]
            clean = col[col != -999]
            avg = np.mean(clean)
            #print(col[col == 0].shape , feature )
            col[col == -999] = avg
    #lighter particles can't be heavier than protons
    for feature in mass_features:
        col = tX[:, feature]
        clean = col[col <= 167]
        avg = np.mean(clean)
        col[col > 167] = avg
    return tX

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly



def standardize_not(tX):
    for feature in range(tX.shape[1]):
        col = tX[:, feature]
        tX[:, feature] = col
        
    return tX

def standardize(tX):
    for feature in range(tX.shape[1]):
        col = tX[:, feature]
        tX[:, feature] = (col - np.mean(col))/np.std(col)
        
    return tX

def min_max(tX):
    for feature in range(tX.shape[1]):
        col = tX[:, feature]
        tX[:, feature] = (col - min(col))/(max(col)-min(col))
        
    return tX  

def mean_normalization(tX):
    for feature in range(tX.shape[1]):
        col = tX[:, feature]
        tX[:, feature] = (col - np.mean(col))/(max(col)-min(col))
        
    return tX
    
def scale_to_unit_length(tX):
    for feature in range(tX.shape[1]):
        col = tX[:, feature]
        tX[:, feature] = col/np.linalg.norm(col)
        
    return tX    

def remove_outliers(tX):
    for feature in range(tX.shape[1]):
        col = tX[:, feature]
        lower = np.mean(col) - (4 * np.std(col))
        higher = np.mean(col) + (4 * np.std(col))
        col[col < lower] = lower
        col[col > higher] = higher
    return tX

def add_column_ones(x):
    return np.hstack([np.expand_dims(np.ones(x.shape[0]), axis=1),x])      

def process(x_tr, x_te, jet_num, features_to_discard, degree_opti, stan ):
    if jet_num in range(4):
        #feature selection
        if (features_to_discard is not None):
            x_tr = np.delete(x_tr, features_to_discard, axis =1)
            x_te =np.delete(x_te, features_to_discard, axis =1)
        #data cleaning
        x_tr=clean(x_tr)
        x_te=clean(x_te)
        if(stan == 0):
            x_tr = standardize_not(x_tr)
            x_te = standardize_not(x_te)
        elif(stan == 1):
            x_tr = standardize(x_tr)
            x_te = standardize(x_te)
        elif(stan == 2):
            x_tr = min_max(x_tr)
            x_te = min_max(x_te)
        elif(stan == 3):
            x_tr = mean_normalization(x_tr)
            x_te = mean_normalization(x_te)
        elif(stan == 4):
            x_tr = scale_to_unit_length(x_tr)
            x_te = scale_to_unit_length(x_te)
            
        
        
        #feature expansion
        x_tr= build_poly(x_tr,degree_opti)
        x_te= build_poly(x_te,degree_opti)
        return x_tr, x_te
    else : 
        print("jet_num not found")
        return  
    