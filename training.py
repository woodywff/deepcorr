import pdb
import yaml
import os
import time
import sys
import numpy as np
import pipeline
from helper import print_red, calc_param_size, ReduceLROnPlateau
from tqdm.notebook import tqdm
from collections import defaultdict, Counter, OrderedDict
import pickle
from utils import accuracy
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import DeepCorrCNN
import shutil

with open('config.yml') as f:
    FLAG_DEBUG = yaml.load(f,Loader=yaml.FullLoader)['FLAG_DEBUG']

class Base:
    '''
    Base class for Training
    cf: config.yml path
    cv_i: Which fold in the cross validation. If cv_i >= n_fold: use all the training dataset.
    '''
    def __init__(self, cf='config.yml', cv_i=0):
        self.cf = cf
        self.cv_i = cv_i
        self._init_config()
        self._init_log()
        self._init_device()
        self._init_dataset()
        
    def _init_config(self):
        with open(self.cf) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        return
    
    def _init_log(self):
        log_path = self.config['data']['log_path']['tf']
        self.train_log = os.path.join(log_path, 'train')
        try:
            os.makedirs(self.train_log)
        except FileExistsError:
            pass

    def _init_device(self):
        '''
        So far we only consider single GPU.
        '''
        seed = self.config['data']['seed']
        np.random.seed(seed)
        tf.random.set_seed(seed)

        try: 
            gpu_check = tf.config.list_physical_devices('GPU')
#             tf.config.experimental.set_memory_growth(gpu_check[0], True)
        except AttributeError: 
            gpu_check = tf.test.is_gpu_available()
        except IndexError:
            gpu_check = False
        if not (gpu_check):
            print_red('We are running on CPU!')
        return
    
    def _init_dataset(self):
        dataset = pipeline.Dataset(cf=self.cf, 
                                   cv_i=self.cv_i) 
        self.train_generator = dataset.train_generator
        self.val_generator = dataset.val_generator
        self.test_generator = dataset.test_generator
        return

class Training(Base):
    '''
    Traing process
    cf: config.yml path
    cv_i: Which fold in the cross validation. If cv_i >= n_fold: use all the training dataset.
    new_lr: if True, check_resume() will not load the saved states of optimizers and lr_schedulers.
    '''
    def __init__(self, cf='config.yml', cv_i=0, new_lr=False):
        super().__init__(cf=cf, cv_i=cv_i)
        self._init_model()
        self.check_resume(new_lr=new_lr)
    
    def _init_model(self):
        self.model = DeepCorrCNN(self.config['train']['conv_filters'],
                                 self.config['train']['dense_layers'], 
                                 self.config['train']['drop_p'])
        self.model(np.random.rand(1,8,300,1).astype('float32'),training=True)
        print('Param size = {:.3f} MB'.format(calc_param_size(self.model.trainable_variables)))
        self.loss = lambda props, y_truth: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=props,labels=y_truth))
        self.optimizer = Adam(self.config['train']['lr']) 
        self.scheduler = ReduceLROnPlateau(self.optimizer, framework='tf')

    def check_resume(self, new_lr=False):
        '''
        Restore saved model, parameters and statistics.
        '''
        checkpoint = tf.train.Checkpoint(model=self.model,
                                         optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                  directory=self.train_log,
                                                  checkpoint_name=self.config['train']['train_last'],
                                                  max_to_keep=1)
        self.train_last = os.path.join(self.train_log, self.config['train']['train_last'])
        self.last_aux = os.path.join(self.train_log, self.config['train']['last_aux'])
        self.train_best = os.path.join(self.train_log, self.config['train']['train_best'])
        self.best_aux = os.path.join(self.train_log, self.config['train']['best_aux'])
        if os.path.exists(self.last_aux):
            checkpoint.restore(self.manager.latest_checkpoint)
            with open(self.last_aux, 'rb') as f:
                state_dicts = pickle.load(f)
            self.epoch = state_dicts['epoch'] + 1
            self.history = state_dicts['history']
            if new_lr:
                del(self.optimizer) # I'm not sure if this is necessary
                self.optimizer = Adam(self.config['train']['lr']) 
            else:
                self.scheduler.load_state_dict(state_dicts['scheduler'])
            self.best_val_loss = state_dicts['best_loss']
        else:
            self.epoch = 0
            self.history = defaultdict(list)
            self.best_val_loss = float('inf')


    def train(self):
        n_epochs = self.config['train']['epochs']
        for epoch in range(n_epochs):
            is_best = False
            loss, acc = self.single_epoch(self.train_generator)
            val_loss, val_acc = self.single_epoch(self.val_generator, 
                                                      training = False, 
                                                      desc = 'Val')
            self.scheduler.step(val_loss)
            self.history['loss'].append(loss)
            self.history['acc'].append(acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_loss < self.best_val_loss:
                is_best = True
                self.best_val_loss = val_loss
            
            # Save what the current epoch ends up with.
            self.manager.save()
            state_dicts = {
                'epoch': self.epoch,
                'history': self.history,
                'scheduler': self.scheduler.state_dict(),
                'best_loss': self.best_val_loss
            }
            with open(self.last_aux, 'wb') as f:
                pickle.dump(state_dicts, f)
                
            if is_best:
                self.model.save_weights(self.train_best)
                shutil.copy(self.last_aux, self.best_aux)
            
            self.epoch += 1
            if self.epoch > n_epochs:
                break
            
            if FLAG_DEBUG and epoch >= 1:
#                 pdb.set_trace()
                break
        return
    
    @tf.function
    def step(self, x, y_truth):
        with tf.GradientTape() as tape:
            props = self.model(x, training=True)
            props = tf.reshape(props, [-1]) # specific for tf.nn.sigmoid_cross_entropy_with_logits
            loss = self.loss(props, y_truth)
#             loss += tf.add_n(self.model.losses) # l2 regularization
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return props, loss
 
    def single_epoch(self, generator, training=True, desc='Training', save_props=False):
        '''
        generator: instance of class Generator.
        training: if True, for training; otherwise, for val or testing.
        desc: desc str uesd for tqdm.
        '''
        n_steps = generator.spe
        sum_loss = 0
        sum_acc = 0
        if save_props:
            saved_props = []
            saved_y_truth = []
        with tqdm(generator.epoch(), total = n_steps,
                  desc = f'{desc} | Epoch {self.epoch}') as pbar:
            for step, (x, y_truth) in enumerate(pbar):
                x = tf.constant(x.astype('float32'))
                y_truth = tf.constant(y_truth.astype('float32'))
                if training:
                    props, loss = self.step(x, y_truth)
                else:
                    props = self.model(x, training=False)
                    props = tf.reshape(props, [-1]) # specific for tf.nn.sigmoid_cross_entropy_with_logits
                    loss = self.loss(props, y_truth)
                sum_loss += loss.numpy()
                acc = accuracy(props.numpy(), y_truth.numpy())
                sum_acc += acc
                if save_props:
                    saved_props = np.hstack([saved_props, props.numpy()])
                    saved_y_truth = np.hstack([saved_y_truth, y_truth.numpy()])
                
                postfix = OrderedDict()
                postfix['Loss'] = round(sum_loss/(step+1), 3)
                postfix['Acc'] = round(sum_acc/(step+1), 3)
                pbar.set_postfix(postfix)
                
                if FLAG_DEBUG and step >= 1:
                    break
        return [round(i/n_steps,3) for i in [sum_loss, sum_acc]] + ([saved_props, saved_y_truth] if save_props else [])
    
    
    def testing(self):
        loss, acc, props, y_truth = self.single_epoch(self.test_generator, 
                                      training = False, 
                                      desc = 'Test',
                                      save_props = True)
        print(f'Testing result: loss = {loss}, accuracy = {acc}')
    
if __name__ == '__main__':
    t = Training()
    t.train()
    t.testing()