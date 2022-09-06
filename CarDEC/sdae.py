from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
from typing import Callable, Iterable, Optional, Tuple, List
import torch
import torch.nn as nn
from time import time
import numpy as np
from .CarDEC_dataloaders import simpleloader,aeloader
from torch.optim import Adam
import os 
from torchsummary import summary

def build_units(
    dimensions: Iterable[int], activation: Optional[torch.nn.Module]
) -> List[torch.nn.Module]:
    """
    Given a list of dimensions and optional activation, return a list of units where each unit is a linear
    layer followed by an activation layer.

    :param dimensions: iterable of dimensions for the chain
    :param activation: activation layer to use e.g. nn.ReLU, set to None to disable
    :return: list of instances of Sequential
    """

    def single_unit(in_dimension: int, out_dimension: int) -> torch.nn.Module:
        unit = [("linear", nn.Linear(in_dimension, out_dimension))]
        if activation is not None:
            unit.append(("activation", activation))
        return nn.Sequential(OrderedDict(unit))

    return [
        single_unit(embedding_dimension, hidden_dimension)
        for embedding_dimension, hidden_dimension in sliding_window(2, dimensions)
    ]


def default_initialise_weight_bias_(
    weight: torch.Tensor, bias: torch.Tensor, gain: float
) -> None:
    """
    Default function to initialise the weights in a the Linear units of the StackedDenoisingAutoEncoder.

    :param weight: weight Tensor of the Linear unit
    :param bias: bias Tensor of the Linear unit
    :param gain: gain for use in initialiser
    :return: None
    """
    nn.init.xavier_uniform_(weight, gain)
    nn.init.constant_(bias, 0)


class StackedDenoisingAutoEncoder(nn.Module):
    def __init__(
        self,
        dimensions: List[int],
        activation: torch.nn.Module = nn.ReLU(),
        final_activation: Optional[torch.nn.Module] = nn.ReLU(),
        weight_init: Callable[
            [torch.Tensor, torch.Tensor, float], None
        ] = default_initialise_weight_bias_,
        gain: float = nn.init.calculate_gain("relu"),
        optim=Adam,
        random_seed = 201809,
        splitseed = 215,
        weights_dir = 'CarDEC Weights',
        extra_dim=None,
    ):
        """
        Autoencoder composed of a symmetric decoder and encoder components accessible via the encoder and decoder
        attributes. The dimensions input is the list of dimensions occurring in a single stack
        e.g. [100, 10, 10, 5] will make the embedding_dimension 100 and the hidden dimension 5, with the
        autoencoder shape [100, 10, 10, 5, 10, 10, 100].

        :param dimensions: list of dimensions occurring in a single stack
        :param activation: activation layer to use for all but final activation, default torch.nn.ReLU
        :param final_activation: final activation layer to use, set to None to disable, default torch.nn.ReLU
        :param weight_init: function for initialising weight and bias via mutation, defaults to default_initialise_weight_bias_
        :param gain: gain parameter to pass to weight_init
        """
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.dimensions = dimensions
        self.embedding_dimension = dimensions[0]
        self.hidden_dimension = dimensions[-1]
        ########################
        self.random_seed = random_seed
        self.splitseed = splitseed
        self.weights_dir = weights_dir
        self.optim = optim
        self.dims=self.dimensions # 
        ########################

        # construct the encoder
        encoder_units = build_units(self.dimensions[:-1], activation)
        encoder_units.extend(
            build_units([self.dimensions[-2], self.dimensions[-1]],final_activation)
        )
        self.encoder = nn.Sequential(*encoder_units)
        # construct the decoder
        if(extra_dim is None):
            decoder_units = build_units(reversed(self.dimensions[1:]), activation)
            decoder_units.extend(
                build_units([self.dimensions[1], self.dimensions[0]]  ,None)
            ) #final output_layers use linear activation
            self.decoder = nn.Sequential(*decoder_units)
        else: # 需要添加额外的维度
            reverse_dim=list(reversed(self.dimensions[1:]))
            reverse_dim[0]=reverse_dim[0]+extra_dim #
            decoder_units = build_units(reverse_dim, activation)
            decoder_units.extend(
                build_units([self.dimensions[1], self.dimensions[0]]  ,None)
            ) #final output_layers use linear activation
            self.decoder = nn.Sequential(*decoder_units)
        # initialise the weights and biases in the layers
        for layer in concat([self.encoder, self.decoder]):
            weight_init(layer[0].weight, layer[0].bias, gain)

    def get_stack(self, index: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Given an index which is in [0, len(self.dimensions) - 2] return the corresponding subautoencoder
        for layer-wise pretraining.

        :param index: subautoencoder index
        :return: tuple of encoder and decoder units
        """
        if (index > len(self.dimensions) - 2) or (index < 0):
            raise ValueError(
                "Requested subautoencoder cannot be constructed, index out of range."
            )
        return self.encoder[index].linear, self.decoder[-(index + 1)].linear

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.decoder(encoded)
    

    def construct(self, summarize = True):
        """ This class method fully initalizes the TensorFlow model.


        Arguments:
        ------------------------------------------------------------------
        - summarize: `bool`, If True, then print a summary of the model architecture.
        """
        
        if summarize:
            print("\n-----------------------CarDEC Architecture-----------------------\n")
            #summary(self)

            print("\n--------------------Encoder Sub-Architecture--------------------\n")
            summary(self.encoder,input_size=(1,self.dims[0]),batch_size=1)
            
            print("\n------------------Base Decoder Sub-Architecture------------------\n")
            summary(self.decoder,input_size=(1,self.dims[-1]),batch_size=1)


    def denoise(self, adata, batch_size = 64):
        """ This class method can be used to denoise gene expression for each cell.


        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars).
        - batch_size: `int`, The batch size used for computing denoised expression.
        
        Returns:
        ------------------------------------------------------------------
        - output: `np.ndarray`, Numpy array of denoised expression of shape (n_obs, n_vars)
        """
        
        input_ds = simpleloader(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'], batch_size)
        
        output = np.zeros((adata.shape[0], self.dims[0]), dtype = 'float32')
        start = 0
        
        for x in input_ds:
            end = start + x.shape[0]
            output[start:end] = self(x).data.numpy()
            start = end
        
        return output
        
    def embed(self, adata, batch_size = 64):
        """ This class method can be used to compute the low-dimension embedding for HVG features. 
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars).
        - batch_size: `int`, The batch size for filling the array of low dimension embeddings.
        
        Returns:
        ------------------------------------------------------------------
        - embedding: `np.ndarray`, Array of shape (n_obs, n_vars) containing the cell HVG embeddings.
        """
        
        input_ds = simpleloader(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'], batch_size)
        
        embedding = np.zeros((adata.shape[0], self.dims[-1]), dtype = 'float32')
        
        start = 0
        for x in input_ds:
            end = start + x.shape[0]
            embedding[start:end] = self.encoder(x).data.numpy()
            start = end
            
        return embedding
    
    def load_autoencoder(self, ):
        """ This class method can be used to load the full model's weights."""
        self=torch.load("./" + self.weights_dir + "/pretrained_autoencoder.pkl")

    def makegenerators(self, adata, val_split, batch_size, splitseed):
        """ This class method creates training and validation data generators for the current input data.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars).
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - batch_size: `int`, The batch size used for training the model.
        - splitseed: `int`, The seed used to split cells between training and validation.
        
        Returns:
        ------------------------------------------------------------------
        - train_dataset: `tf.data.Dataset`, Dataset that returns training examples.
        - val_dataset: `tf.data.Dataset`, Dataset that returns validation examples.
        """
        
        return aeloader(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'], adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'], val_frac = val_split, batch_size = batch_size, splitseed = splitseed)
    
    def trainSaeModel(self,adata, num_epochs = 2000, batch_size = 64, val_split = 0.1, lr = 1e-03, decay_factor = 1/3,
              patience_LR = 3, patience_ES = 9, save_fullmodel = True):
        """ This class method can be used to train the SAE.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars).
        - num_epochs: `int`, The maximum number of epochs allowed to train the full model. In practice, the model will halt training long before hitting this limit.
        - batch_size: `int`, The batch size used for training the full model.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - lr: `float`, The learning rate for training the full model.
        - decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not decreasing.
        - patience_LR: `int`, The number of epochs tolerated before decaying the learning rate during which the validation loss fails to decrease.
        - patience_ES: `int`, The number of epochs tolerated before stopping training during which the validation loss fails to decrease.
        - save_fullmodel: `bool`, If True, save the full model's weights, not just the encoder.
        """
        
        #tf.keras.backend.clear_session() # 首先这个地方就涉及到了generator的问题了，因为这个是涉及到数据的问题的
        
        dataset = self.makegenerators(adata, val_split = 0.1, batch_size = batch_size, splitseed = self.splitseed)
        
        
        counter_LR = 0
        counter_ES = 0
        best_loss = np.inf
        
        self.optimizer=self.optim(self.parameters(),lr=lr)
        loss_function = nn.MSELoss()
        total_start = time()
        for epoch in range(num_epochs):
            epoch_start = time()
            #epoch_loss_avg = tf.keras.metrics.Mean()
            #epoch_loss_avg_val = tf.keras.metrics.Mean()
            epoch_loss_avg =[]
            epoch_loss_avg_val=[]
            # Training loop - using batches of batch_size
            self.train()
            for x, target in dataset(val = False):
                output=self(x)
                loss = loss_function(output,target)
                self.optimizer.zero_grad() #
                loss.backward() #
                self.optimizer.step() # 
                epoch_loss_avg.append(loss.item())
                #loss_value, grads = grad(self, x, target, MSEloss)
                #self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                #epoch_loss_avg(loss_value)  # Add current batch loss
            self.eval()
            # Validation Loop
            with torch.no_grad():
                for x, target in dataset(val = True):
                    output = self(x)
                    loss = loss_function(output,target)
                    epoch_loss_avg_val.append(loss.item())
                    #loss_value = MSEloss(target, output)
                    #epoch_loss_avg_val(loss_value)
            
            #current_loss_val = epoch_loss_avg_val.result()
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
        
        #self.save_weights("./" + self.weights_dir + "/pretrained_autoencoder_weights", save_format='tf')
        #self.encoder.save_weights("./" + self.weights_dir + "/pretrained_encoder_weights", save_format='tf')
        torch.save(self,"./" + self.weights_dir + "/pretrained_autoencoder.pkl")
        print('\nTraining Completed')
        print("Total training time: " + str(total_time) + " seconds")
