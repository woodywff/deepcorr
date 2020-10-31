import os
import gzip
import numpy as np

def load_fmnist(path, is_train=True):
    '''
    Rerutn: images.shape = (num of imgs,28*28)
    ATTENTION: the returned ndarray is not writable.
    '''
    kind = 'train' if is_train else 't10k'
    labels_path = os.path.join(path, '{}-labels-idx1-ubyte.gz'.format(kind))
    images_path = os.path.join(path, '{}-images-idx3-ubyte.gz'.format(kind))
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels

def calc_param_size(params):
    '''
    Show the memory cost of model.parameters, in MB. 
    It works for float32 parameters.
    params: for tensorflow it's a list, for pytorch and paddlepaddle it's a generator.
    '''
    return np.sum(np.prod(p.shape) for p in params)*4e-6

def print_red(something):
    print("\033[1;31m{}\033[0m".format(something))
    return

class ReduceLROnPlateau():
    '''
    opt: optimizer
    factor (float): Factor by which the learning rate will be
        reduced. new_lr = lr * factor.
    patience (int): Number of epochs with no improvement after
        which learning rate will be reduced. For example, if
        patience = 2, then we will ignore the first 2 epochs
        with no improvement, and will only decrease the LR after the
        3rd epoch if the loss still hasn't improved then.
    verbose (bool): If True, prints a message to stdout for
        each update.
    framework: 'pp' for paddlepaddle, 'tf' for tensorflow.
    '''
    def __init__(self, opt, patience=10, factor=0.5, verbose=True, framework='pp'):
        self.opt = opt
        self.patience = patience
        self.factor = factor
        self.verbose = verbose
        self.record = float('inf')
        self.count = 0
        self.framework = framework
        
    def step(self, loss):
        if loss >= self.record:
            self.count += 1
            if self.count == self.patience:
                if self.framework == 'pp':
                    self.opt._learning_rate *= self.factor
                elif self.framework == 'tf':
                    self.opt._set_hyper('learning_rate', self.factor * self.opt.get_config()['learning_rate'])
                else:
                    raise NotImplementedError
                self.count = 0
                self.record = loss
        else:
            self.count = 0
            self.record = loss
        return
    
    def state_dict(self):
        if self.framework == 'pp':
            lr = self.opt._learning_rate
        elif self.framework == 'tf':
            lr = self.opt.get_config()['learning_rate']
        else:
            raise NotImplementedError
        return {'record':self.record,
                'count': self.count,
                'lr': lr,
                'patience': self.patience,
                'factor': self.factor}
    
    def load_state_dict(self, state_dict, full_load=False):
        '''
        state_dict (dict): State_dict to be recovered.
        full_load (bool): If True, recovers the self.patience and self.factor.
        '''
        self.record = state_dict['record']
        self.count = state_dict['count']
        if self.framework == 'pp':
            self.opt._learning_rate = state_dict['lr']
        elif self.framework == 'tf':
            self.opt._set_hyper('learning_rate', state_dict['lr'])
        else:
            raise NotImplementedError
        if full_load:
            self.patience = state_dict['patience']
            self.factor = state_dict['factor']
        return
    