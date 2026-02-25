import numpy as np

X = np.array([
    [150, 70],
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
])

y = np.array([0,1,1,0,0,1,1,0,1,0])

n = [2, 3, 3, 1]

W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)

W = np.array([0,W1,W2,W3], dtype=object)
b = np.array([0,b1,b2,b3], dtype=object)

def prepare_data(X, y, n):
    # X is a np.array with the training data (m x n[0])
    # y is a np.array the associated scores (m x n[L])
    # n is an array containing the number of neurons per layer
    
    m = len(X)
    L = len(n) - 1
    
    A0 = X.T # Our input layer
    Y = y.reshape(n[L], m) # Shape m * n[L]
    
    return A0, Y, m
    
def g(z):
  return 1 / (1 + np.exp(-1 * z))

def feed_forward(A0, W, b):
    # A0 is the input layer
    # W is a np ragged nested array of weights for each layer dtype=object
    # b is a np ragged array of biases for each layer dtype=object
    
    cache = []
    cache.append(A0)
    
    for i in range(1, len(W)):
        Z = W[i] @ cache[i-1] + b[i]
        A = g(Z)
        cache.append(A)
        
    return cache[-1], cache


def cost(y_hat, y):
    # y_hat is a n[L] x m matrix
    # y is a n[L] x m matrix
    
    losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )

    m = y_hat.reshape(-1).shape[0]

    summed_losses = (1 / m) * np.sum(losses, axis=1)

    return np.sum(summed_losses)

# Main

A0,Y,m = prepare_data(X, y, n)
A3, cache = feed_forward(A0, W, b)