import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N, D = X.shape

    for i in xrange(N):

        # forward pass
        score = np.dot(X[i, :], W)
        exp = np.exp(score)
        expsum = np.sum(exp)
        prob = exp / expsum
        loss += -np.log(prob[y[i]])

        # back pass
        dprob = np.zeros_like(prob)
        dprob[y[i]] = -1.0 / prob[y[i]]
        dexpsum = -1.0 / expsum**2 * np.sum(exp * dprob)
        dexp = 1.0 / expsum * dprob + np.ones_like(exp) * dexpsum
        dscore = exp * dexp
        dW += np.outer(X[i, :], dscore)

    loss /= N
    dW /= N

    # regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N, D = X.shape
    _, C = W.shape

    # forward pass
    score = np.dot(X, W) # N,D x D,C = N,C
    score -= np.max(score, axis=1).reshape(N,1) # stabilize f
    exp = np.exp(score) # N,C
    expsum = np.sum(exp, axis=1) # N,1
    prob = exp / expsum.reshape(N,1) # N,C
    loss = -np.log(prob[np.arange(N), y]) # N,1

    # back pass
    dproby = -1. / prob[np.arange(N), y] # N,1
    dexpsum = -1. / expsum**2 * exp[np.arange(N), y] * dproby # N,1
    dexp = np.repeat(dexpsum.reshape(N,1), C, axis=1) # N,C
    dexp[np.arange(N), y] += 1. / expsum * dproby # N,
    dscore = exp * dexp # N,C
    dW = np.dot(X.T, dscore) # D,C

    loss = np.sum(loss) / N
    dW /= N

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

def softmax_loss_alt(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N, D = X.shape
    _, C = W.shape

    # forward pass
    score = np.dot(X, W) # N,D x D,C = N,C
    score -= np.max(score, axis=1).reshape(N, 1) # stabilize f
    exp = np.exp(score) # N,C
    expsum = np.sum(exp, axis=1).reshape(N, 1) # N,1
    prob = exp / expsum # N,C
    loss = -np.log(prob[np.arange(N), y]) # N,1

    # back pass
    dscore = prob # N,C
    dscore[np.arange(N), y] -= 1
    dW = np.dot(X.T, dscore) # D,C

    loss = np.sum(loss) / N
    dW /= N

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

