import numpy as np
from costs import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    
    Parameters
    ----------
    y : numpy.ndarray
       The predictions.
    
    tx : numpy.ndarray
       The data points.
    
    initial_w : numpy.ndarray
        Inital weight vector.
    
    max_iters : int
        The number of iterations of gradient descent.
    
    gamma : float
       The step size.

    Returns
    -------
    w, loss 
        the weight vector w,  and the loss
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss= compute_mse(y,tx,w)
        w = w- gamma*compute_gradient(y,tx,w)
        ws.append(w)
        losses.append(loss)
    return ws[-1], losses[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    
    Parameters
    ----------
    y : numpy.ndarray
       The predictions.
    
    tx : numpy.ndarray
       The data points.
    
    initial_w : numpy.ndarray
        Inital weight vector.
    
    max_iters : int
        The number of iterations of gradient descent.
    
    gamma : float
       The step size.

    Returns
    -------
    w, loss 
        the weight vector w,  and the loss
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n in range(max_iters):
        for b_y,b_x in batch_iter(y,tx, batch_size =1):
            loss =compute_mse(b_y,b_x,w)
            grad = compute_stoch_gradient(b_y,b_x,w)
            w = w-gamma*grad 
            ws.append(w)
            losses.append(loss)
    return ws[-1], losses[-1]

def least_squares(y, tx):
    """
    Least squares regression using normal equations
    
    Parameters
    ----------
    y : numpy.ndarray
       The predictions.
    
    tx : numpy.ndarray
       The data points.
    
    Returns
    -------
    w, loss 
        the weight vector w,  and the loss
    """
    a=tx.T@tx
    b=tx.T@y
    w=np.linalg.solve(a, b)
    loss= compute_mse(y,tx,w)
    return w , loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    
    Parameters
    ----------
    y : numpy.ndarray
       The predictions.
    
    tx : numpy.ndarray
       The data points.
       
    lambda_ : float
       The regularization hyperparameter

    Returns
    -------
    w, loss 
        the weight vector w,  and the loss
    """
    temp_1=tx.T@tx
    temp_2= 2*(tx.shape[0])*lambda_*np.identity(tx.shape[1])
    a= temp_1 + temp_2
    b=tx.T@y
    w=np.linalg.solve(a, b)
    loss = compute_rmse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    
    Parameters
    ----------
    y : numpy.ndarray
       The predictions.
    
    tx : numpy.ndarray
       The data points.
    
    initial_w : numpy.ndarray
        Inital weight vector.
    
    max_iters : int
        The number of iterations of gradient descent.
    
    gamma : float
       The step size.

    Returns
    -------
    w, loss 
        the weight vector w,  and the loss
    """
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    return w, losses[-1]


def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    
    Parameters
    ----------
    y : numpy.ndarray
       The predictions.
    
    tx : numpy.ndarray
       The data points.
    
    initial_w : numpy.ndarray
        Inital weight vector.
    
    max_iters : int
        The number of iterations of gradient descent.
    
    gamma : float
       The step size.
    
    lambda_ : float
        The regularization hyperparameter
    
    Returns
    -------
    w, loss 
        the weight vector w,  and the loss
    """
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    return w, losses[-1]


"""Gradient Descent"""
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y- (tx@w)
    grad= (-1/len(y))*(tx.T)@e
    return grad
   


"""Stochastic Gradient Descent"""
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y- (tx@w)
    grad= (-1/len(y))*(tx.T)@e
    
    return grad
#########################

def sigmoid(x):
    "Numerically stable sigmoid function."
    x[x > 100] = 100
    z = np.exp(x)
    return z / (1 + z)


def calculate_loss(y, tx, w):
    """Numerically stable way to compute the cost by negative log likelihood."""
    val = sigmoid(tx@w)
    val[val == 0] = 10**-(323)
    lhs = y*np.log(val)
    val2 = 1-val
    val2[val2 == 0] = 10**-(323)
    rhs = (1-y)*np.log(val2)
    return -np.sum(lhs+rhs)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w) - y)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma*grad
    return w, loss

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss = calculate_loss(y, tx, w) + lambda_*np.squeeze(w.T@w)
    grad = calculate_gradient(y, tx, w) + 2*lambda_ * w
    return loss, grad

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad =  penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma*grad
    return loss, w

def calculate_penalized_loss(y, tx, w, lambda_):
     return calculate_loss(y, tx, w) + lambda_*np.squeeze(w.T@w)
     
    




