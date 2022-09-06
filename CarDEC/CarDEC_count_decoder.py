
from CarDEC.loss import NBLoss
from .CarDEC_utils import build_dir
from .CarDEC_dataloaders import countloader, tupleloader

#from tensorflow.keras import Model, Sequential
from torch.nn import Sequential
import torch.nn as nn
from torch.nn import ReLU,Tanh
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
import torch
from typing import Any, Callable, Optional
import math
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.optim import SGD
import torch.nn.functional as F
#from loss_optimization import TotalLoss
import pickle

from time import time

import random
import numpy as np
from scipy.stats import zscore
import os
from .loss import MeanAct,DispAct
from .sdae import build_units,default_initialise_weight_bias_

from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
from typing import Callable, Iterable, Optional, Tuple, List
from torchsummary import summary


class count_model(nn.Module):
    def __init__(self, dims, act = nn.ReLU(), random_seed = 201809, splitseed = 215, optim = Adam,
             weights_dir = 'CarDEC Count Weights', n_features = 32, mode = 'HVG'):
        """ This class method initializes the count model.


        Arguments:
        ------------------------------------------------------------------
        - dims: `list`, the number of output features for each layer of the model. The length of the list determines the
        number of layers.
        - act: `str`, The activation function used for the intermediate layers of CarDEC, other than the bottleneck layer.
        - random_seed: `int`, The seed used for random weight intialization.
        - splitseed: `int`, The seed used to split cells between training and validation. Should be consistent between
        iterations to ensure the same cells are always used for validation.
        - optimizer: `tensorflow.python.keras.optimizer_v2`, An instance of a TensorFlow optimizer.
        - weights_dir: `str`, the path in which to save the weights of the CarDEC model.
        - n_features: `int`, the number of input features.
        - mode: `str`, String identifying whether HVGs or LVGs are being modeled.
        """
        super(count_model, self).__init__()        
        self.mode = mode
        self.name_ = mode + " Count"
        
        if mode == 'HVG':
            self.embed_name = 'embedding'
        else:
            self.embed_name = 'LVG embedding'
            dims[-1]=n_features # 如果是LVG，这个是需要
        
        self.weights_dir = weights_dir
        
        self.dims = dims
        self.dimensions=self.dims
        n_stacks = len(dims) - 1
        
        self.optim = optim
        self.random_seed = random_seed
        self.splitseed = splitseed
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        #tf.random.set_seed(random_seed)
        
        self.activation = act
   
        #encoder_units = build_units(self.dimensions[:-1], self.activation)
        # encoder_units.extend(
        #     build_units([self.dimensions[-2], self.dimensions[-1]],final_activation)
        # ) 
        decoder_units = build_units(reversed(self.dimensions[1:]), self.activation)

        self.base = nn.Sequential(*decoder_units)

        self.mean_layer = nn.Sequential(nn.Linear(self.dimensions[1], self.dimensions[0]),MeanAct())

        self.disp_layer = nn.Sequential(nn.Linear(self.dimensions[1], self.dimensions[0]),DispAct())
        
        #self.rescale = Lambda(lambda l: tf.matmul(tf.linalg.diag(l[0]), l[1]), name = 'sf scaling')
        
        build_dir(self.weights_dir)
        
        #self.construct(n_features, self.name_)
         
        weight_init=default_initialise_weight_bias_
        gain=1.0
        for layer in self.base:
            weight_init(layer[0].weight, layer[0].bias, gain)
        
        weight_init(self.mean_layer[0].weight,self.mean_layer[0].bias,gain)# self.mean_layer is different form self.base
        weight_init(self.disp_layer[0].weight,self.disp_layer[0].bias,gain)

        self.construct(n_features, self.name_)

    def forward(self, x):
        """ This is the forward pass of the model.
        

        ***Inputs***
            - x: `tf.Tensor`, an input tensor of shape (b, p)
            - s: `tf.Tensor`, and input tensor of shape (b, ) containing the size factor for each cell
            
        ***Outputs***
            - mean: `tf.Tensor`, A (b, p_gene) tensor of negative binomial means for each cell, gene.
            - disp: `tf.Tensor`, A (b, p_gene) tensor of negative binomial dispersions for each cell, gene.
        """
        
        x = self.base(x)
        
        disp = self.disp_layer(x)
        mean = self.mean_layer(x)
        #mean = self.rescale([s, mean])
                        
        return mean, disp
        
    def load_model(self, ):
        """ This class method can be used to load the model's weights."""
            
        #tf.keras.backend.clear_session()
        
        self.load_weights(os.path.join(self.weights_dir, "countmodel_weights_" + self.name_)).expect_partial()
        
    def construct(self, n_features,name,summarize = True):
        """ This class method fully initalizes the TensorFlow model.


        Arguments:
        ------------------------------------------------------------------
        - n_features: `int`, the number of input features.
        - name: `str`, Model name (to distinguish HVG and LVG models).
        - summarize: `bool`, If True, then print a summary of the model architecture.
        """
        
        #x = [tf.zeros(shape = (1, n_features), dtype='float32'), tf.ones(shape = (1,), dtype='float32')]
        
        if summarize:
            print("----------Count Model " + name + " Architecture----------")
            summary(self.mean_layer,input_size=(1,self.dimensions[1]),batch_size=1)
            summary(self.disp_layer,input_size=(1,self.dimensions[1]),batch_size=1)
            #print("\n----------Base Sub-Architecture----------")
            summary(self.base,input_size=(1,n_features),batch_size=1)
        
    def denoise(self, adata, keep_dispersion = False, batch_size = 64):
        """ This class method can be used to denoise gene expression for each cell on the count scale.


        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars). Rows correspond
        to cells and columns to genes.
        - keep_dispersion: `bool`, If True, also return the dispersion for each gene, cell (added as a layer to adata)/
        - batch_size: `int`, The batch size used for computing denoised expression.
        
        Returns:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars). Negative binomial means (and optionally 
        dispersions) added as layers.
        """
        
        input_ds = tupleloader(adata.obsm[self.embed_name], adata.obs['size factors'], batch_size = batch_size)
        
        if "denoised counts" not in list(adata.layers):
            adata.layers["denoised counts"] = np.zeros(adata.shape, dtype = 'float32')
        
        type_indices = adata.var['Variance Type'] == self.mode
        
        if not keep_dispersion:
            start = 0
            for x in input_ds:
                end = start + x[0].shape[0]
                adata.layers["denoised counts"][start:end, type_indices] = self(x[0])[0].data.numpy()
                start = end
                
        else:
            if "dispersion" not in list(adata.layers):
                adata.layers["dispersion"] = np.zeros(adata.shape, dtype = 'float32')
                
            start = 0
            for x in input_ds:
                end = start + x[0].shape[0]
                batch_output = self(*x)
                adata.layers["denoised counts"][start:end, type_indices] = batch_output[0].numpy()
                adata.layers["dispersion"][start:end, type_indices] = batch_output[1].numpy()
                start = end
            
    def makegenerators(self, adata, val_split, batch_size, splitseed):
        """ This class method creates training and validation data generators for the current input data.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond
        to cells and columns to genes.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - batch_size: `int`, The batch size used for training the model.
        - splitseed: `int`, The seed used to split cells between training and validation. Should be consistent between
        iterations to ensure the same cells are always used for validation.
        
        Returns:
        ------------------------------------------------------------------
        - train_dataset: `tf.data.Dataset`, Dataset that returns training examples.
        - val_dataset: `tf.data.Dataset`, Dataset that returns validation examples.
        """
        
        return countloader(adata.obsm[self.embed_name], adata.X[:, adata.var['Variance Type'] == self.mode], adata.obs['size factors'], 
                           val_split, batch_size, splitseed)
    
    def trainModel(self, adata, num_epochs = 2000, batch_size = 64, val_split = 0.1, lr = 1e-03, decay_factor = 1/3,
              patience_LR = 3, patience_ES = 9):
        """ This class method can be used to train the SAE.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars). Rows correspond
        to cells and columns to genes.
        - num_epochs: `int`, The maximum number of epochs allowed to train the full model. In practice, the model will halt
        training long before hitting this limit.
        - batch_size: `int`, The batch size used for training the full model.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - lr: `float`, The learning rate for training the full model.
        - decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not
        decreasing.
        - patience_LR: `int`, The number of epochs tolerated before decaying the learning rate during which the
        validation loss fails to decrease.
        - patience_ES: `int`, The number of epochs tolerated before stopping training during which the validation loss fails to
        decrease.
        """
        
        #tf.keras.backend.clear_session()
                
        nbloss = NBLoss()
        
        dataset = self.makegenerators(adata, val_split = 0.1, batch_size = batch_size, splitseed = self.splitseed)
        
        counter_LR = 0
        counter_ES = 0
        best_loss = np.inf

        self.optimizer=self.optim(self.parameters(),lr=lr)
        
        total_start = time()
        
        for epoch in range(num_epochs):
            epoch_start = time()
            
            epoch_loss_avg = []
            epoch_loss_avg_val = []
            self.train()
            # Training loop - using batches of batch_size
            for x,target in dataset(val = False): # x[0]:scaled, x[1]:sf_factor, target:X_raw
                mean_tensor,disp_tensor=self(x[0])
                loss=nbloss(target, mean=mean_tensor, disp=disp_tensor, scale_factor=x[1])
                self.optimizer.zero_grad() #
                loss.backward() #
                self.optimizer.step() # 
                epoch_loss_avg.append(loss.item())
                #print("loss={}".format(loss.item()))

            self.eval()
            # Validation Loop
            with torch.no_grad():
                for x, target in dataset(val = True):
                    mean_tensor,disp_tensor=self(x[0])
                    loss=nbloss(target, mean=mean_tensor, disp=disp_tensor, scale_factor=x[1])
                    epoch_loss_avg_val.append(loss.item())

            current_loss_val=np.mean(np.array(epoch_loss_avg_val))

            epoch_time = round(time() - epoch_start, 1)
            print("Epoch {:03d}: Training Loss: {:.3f}, Validation Loss: {:.3f}, Time: {:.1f} s".format(epoch, np.mean(np.array(epoch_loss_avg)),np.mean(np.array(epoch_loss_avg_val)) , epoch_time))
            
            if(current_loss_val + 10**(-3) < best_loss):
                counter_LR = 0
                counter_ES = 0
                best_loss = current_loss_val
            else:
                counter_LR = counter_LR + 1
                counter_ES = counter_ES + 1

            if patience_ES <= counter_ES:
                break

            if patience_LR <= counter_LR:
                self.optimizer.param_groups[0]["lr"]= self.optimizer.param_groups[0]["lr"] * decay_factor
                counter_LR = 0
                print("\nDecaying Learning Rate to: " + str(self.optimizer.param_groups[0]["lr"]))
                
            # End epoch
        
        total_time = round(time() - total_start, 2)
        
        if not os.path.isdir("./" + self.weights_dir):
            os.mkdir("./" + self.weights_dir)
        
        #self.save_weights(os.path.join(self.weights_dir, "countmodel_weights_" + self.name_), save_format='tf')
        torch.save(self,"./" + self.weights_dir + "/countmodel.pkl")        
        print('\nTraining Completed')
        print("Total training time: " + str(total_time) + " seconds")

