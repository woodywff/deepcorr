import numpy as np

def accuracy(logits, y_truth):
    '''
    logits.shape == y_truth.shape == [batch_size,]
    logits: output from sigmoid.
    '''
    batch_size = y_truth.shape[0]
    return np.sum((logits >= 0.5) == y_truth)/batch_size