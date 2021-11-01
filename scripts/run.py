# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from costs import *
from process import *

from proj1_helpers import *

DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


DATA_TEST_PATH = '../data/test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#***********CONST*************#
to_discard = ([[4,5,6,12,22,23,24,25,26,27,28,29],[4,5,6,12,22,26,27,28],[22], [22]])
degrees_opti = np.array([7,7,7,7])
lambdas_opti = np.array([10e-8, 10e-8,  10e-8, 10e-8])

def masks(X_tr,X_te,jet_num):
    if (jet_num == 0):
        return (X_tr[:, 22] == jet_num), (X_te[:, 22] == jet_num)
    elif (jet_num == 1):
        return (X_tr[:, 22] == jet_num), (X_te[:, 22] == jet_num)
    elif np.logical_or(jet_num == 2,jet_num== 3 ):
        return np.logical_or(X_tr[:, 22] == 2,X_tr[:, 22] == 3 ) , np.logical_or(X_te[:, 22] == 2,X_te[:, 22] == 3 ) 
    else :
        print("invalid jet_num")
        return

def maskss(X_tr,X_te,jet_num):
    if (jet_num == 0):
        return (X_tr[:, 22] == jet_num), (X_te[:, 22] == jet_num)
    elif (jet_num == 1):
        return (X_tr[:, 22] == jet_num), (X_te[:, 22] == jet_num)
    elif (jet_num == 2):
        return (X_tr[:, 22] == jet_num), (X_te[:, 22] == jet_num)
    elif (jet_num == 3):
        return (X_tr[:, 22] == jet_num), (X_te[:, 22] == jet_num)
        
    else :
        print("invalid jet_num")
        return   
    
        
#***********MAIN*************#
def main(X_tr, Y_tr, X_te ):

    #init final predictions vector
    y_fin = np.zeros(X_te.shape[0])
    #tX_test

    for jet_num in range(4):

        #print(jet_num)
        #initialize the mask according to jet_num
        mask_tr, mask_te = maskss(X_tr,X_te,jet_num)


        #param
        features_to_discard = to_discard[jet_num]
        degree = degrees_opti[jet_num]
        lambda_ = lambdas_opti[jet_num]

        #print(features_to_discard,degree, lambda_)
        #filter on jet_num
        ## for training
        x_tr = X_tr[mask_tr]
        y_tr = Y_tr[mask_tr]
        ## for Testing 
        x_te = X_te[mask_te]

        #process
        tx_tr,tx_te = process(x_tr, x_te, jet_num, features_to_discard, degree, stan = 1 )

        #train
        w_star, loss_tr = least_squares(y_tr,tx_tr)
        #w_star, loss_tr = least_squares_GD(y_tr, tx_tr, np.zeros(tx_tr.shape[1]), 100, 10e-7)

        #store in y_pred
        pred = predict_labels(w_star,tx_te)
        y_fin[mask_te] = pred
    print("done")
    return y_fin

#degree = 7

#Simulation by splitting the data  
ratio =0.75
x_tr,y_tr,x_te,y_te = split_data(tX, y, ratio, seed=22)
y_fin = main(x_tr,y_tr,x_te)

#See Accuracy
print(compute_accuracy(y_fin,y_te),"%")

# removing zeros only in last column and -999 in the rest: 
#no stan: 82.6896
#z-score: 82.6208
# min-max normalization: 77.3344
# mean normalization: 76.3984
# scaling to unit length: 71.7872

#data cleaning (no standardization):
# replacing by random number
# replacing by mean instead of median: 82.7872
# removing invalid mass: 82.7632, 
# removing 0 in feature 29 and -999 in the rest: 82.6896
# without cleaning 82.304
# replacing with 

#82.88

#For Submisison
OUTPUT_PATH = '../data/predictions_final.csv' # TODO: fill in desired name of output file for submission
y_pred = main(tX,y,tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)