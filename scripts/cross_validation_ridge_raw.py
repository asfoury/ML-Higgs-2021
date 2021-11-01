import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *
from implementations import *
from costs import *
from process import *

def cross_validation_ridge_regression(y, x, k_indices, k, lambda_, jet_num, features, degree, stan):
    """return the loss of ridge regression."""
    
    #ridge_regression(y_jet_tr,x_jet_tr,lambda_)
    
    mask = np.ones(k_indices.shape[0],dtype=bool)
    mask[k] = 0
    #print(mask)
    #print(k_indices.shape)
    
    train_indices = k_indices[mask].reshape(-1)
    test_indices = k_indices[k].reshape(-1)
    
    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]
    
    x_train, x_test = process(x_train, x_test, jet_num  , features, degree, stan)

    #compute w_star, loss of both training and test set 
    w_star, loss_tr = ridge_regression(y_train, x_train, lambda_)
    loss_te = compute_rmse(y_test, x_test, w_star)
    
    
    
    return loss_tr, loss_te



def cross_validation_demo_ridge_regression(x_tr,y_tr, jet_num, features, degree, stan):
    seed = 12
    k_fold = 4
    lambdas = np.logspace(-8, -3, 20)
    # split data in k fold
    k_indices = build_k_indices(y_tr, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    counter = 0
    for l in lambdas: 
        #print(counter)
        counter+=1
        rmse_tr_temp = []
        rmse_te_temp = []
        for i in range(k_fold-1): # 4
            loss_tr, loss_te = cross_validation_ridge_regression(y_tr,x_tr,k_indices,i,l,jet_num,features,degree,stan)
            rmse_tr_temp.append(loss_tr)
            #print("for k =",i, " loss_tr = ",loss_tr)
            rmse_te_temp.append(loss_te)
            #print("for k =",i,"and loss_te = ",loss_te)
        rmse_tr.append(np.mean(rmse_tr_temp))
        rmse_te.append(np.mean(rmse_te_temp))
        
    cross_validation_visualization(lambdas,rmse_tr, rmse_te)

#*************CROSS_VALIDATION_on_ACCURACY***************#



def cross_validation_ridge_regression_onACC(y, x, k_indices, k, lambda_, jet_num, features, degree, stan):
    """return the loss of ridge regression."""
    
    #ridge_regression(y_jet_tr,x_jet_tr,lambda_)
    mask = np.ones(k_indices.shape[0],dtype=bool)
    mask[k] = 0
    train_indices = k_indices[mask].reshape(-1)
    test_indices = k_indices[k].reshape(-1)
    
    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]
    

    x_train, x_test =process(x_train, x_test,jet_num,features,degree, stan)
    

    w_star,_ = ridge_regression(y_train, x_train, lambda_)

    
    acc_tr= compute_accuracy(predict_labels(w_star,x_train),y_train)
    acc_te= compute_accuracy(predict_labels(w_star,x_test),y_test)
    
    return acc_tr, acc_te



def cross_validation_demo_ridge_regression_onACC(x_tr,y_tr, jet_num, features, degree, stan):
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-9, -1, 20)
    # split data in k fold
    k_indices = build_k_indices(y_tr, k_fold, seed)
    # define lists to store the loss of training data and test data
    acc_tr = []
    acc_te = []
    counter = 1
    for l in lambdas: 
        
        counter+=1
        #rmse_tr_temp = []
        #rmse_te_temp = []
        acc_tr_temp = []
        acc_te_temp = []
        
        for i in range(k_fold): # 4
            acc_tr_val,acc_te_val=cross_validation_ridge_regression_onACC(y_tr,x_tr,k_indices,i,l,jet_num,features,degree,stan)
            acc_tr_temp.append(acc_tr_val)
            acc_te_temp.append(acc_te_val)
        
        acc_tr.append(np.mean(acc_tr_temp)) 
        acc_te.append(np.mean(acc_te_temp))
        
        
    cross_validation_visualization(lambdas, acc_tr, acc_te)

    

