import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#(target, denoised_output, p, cluster_output_q, LVG_target = None, aeloss_fun = None, clust_weight = 1.):
class TotalLoss(nn.Module):
    def __init__(self, clust_weight=1.0):
        super(TotalLoss, self).__init__()
        self.clust_weight =clust_weight 
        self.KLD= nn.KLDivLoss(reduction="batchmean")
    def forward(self, target, denoised_output, p, cluster_output_q, LVG_target = None, aeloss_fun = None):
        if aeloss_fun is not None:
            aeloss_HVG = aeloss_fun(target, denoised_output['HVG_denoised'])
            if LVG_target is not None:
                aeloss_LVG = aeloss_fun(LVG_target, denoised_output['LVG_denoised'])
                aeloss = 0.5*(aeloss_LVG + aeloss_HVG)
            else:
                aeloss = 1. * aeloss_HVG
        else:
            aeloss = 0.
        
        net_loss = self.clust_weight * self.KLD(cluster_output_q.log(),p) + (2. - self.clust_weight) * aeloss
    
        return net_loss,aeloss


class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=torch.FloatTensor([1.0])):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        result = torch.mean(nb_final)
        return result


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


# class TotalLoss():
#      super(TotalLoss):

# 自定义loss function

# def grad_MainModel(model, input_, target, target_p, total_loss, LVG_target = None, aeloss_fun = None, clust_weight = 1.):
#     """Function to do a backprop update to the main CarDEC model for a minibatch.
    
    
#     Arguments:
#     ------------------------------------------------------------------
#     - model: `tensorflow.keras.Model`, The main CarDEC model.
#     - input_: `list`, A list containing the input HVG and (optionally) LVG expression tensors of the minibatch for the CarDEC model.
#     - target: `tf.Tensor`, Tensor containing the reconstruction target of the minibatch for the HVGs.
#     - target_p: `tf.Tensor`, Tensor containing cluster membership probability targets for the minibatch.
#     - total_loss: `function`, Function to compute the loss for the main CarDEC model for a minibatch.
#     - LVG_target: `tf.Tensor` (Optional), Tensor containing the reconstruction target of the minibatch for the LVGs.
#     - aeloss_fun: `function`, Function to compute reconstruction loss.
#     - clust_weight: `float`, A float between 0 and 2 balancing clustering and reconstruction losses.
    
#     Returns:
#     ------------------------------------------------------------------
#     - loss_value: `tf.Tensor`: The loss computed for the minibatch.
#     - gradients: `a list of Tensors`: Gradients to update the model weights.
#     """
    
#     with tf.GradientTape() as tape:
#         denoised_output, cluster_output = model(*input_)
#         loss_value, aeloss = total_loss(target, denoised_output, target_p, cluster_output, 
#                                 LVG_target, aeloss_fun, clust_weight)
        
#     return loss_value, tape.gradient(loss_value, model.trainable_variables)


# def total_loss(target, denoised_output, p, cluster_output_q, LVG_target = None, aeloss_fun = None, clust_weight = 1.):
#     """Function to compute the loss for the main CarDEC model for a minibatch.
    
    
#     Arguments:
#     ------------------------------------------------------------------
#     - target: `tf.Tensor`, Tensor containing the reconstruction target of the minibatch for the HVGs.
#     - denoised_output: `dict`, Dictionary containing the output tensors from the CarDEC main model's forward pass.
#     - p: `tf.Tensor`, Tensor of shape (n_obs, n_cluster) containing cluster membership probability targets for the minibatch.
#     - cluster_output_q: `tf.Tensor`, Tensor of shape (n_obs, n_cluster) containing predicted cluster membership probabilities
#     for each cell.
#     - LVG_target: `tf.Tensor` (Optional), Tensor containing the reconstruction target of the minibatch for the LVGs.
#     - aeloss_fun: `function`, Function to compute reconstruction loss.
#     - clust_weight: `float`, A float between 0 and 2 balancing clustering and reconstruction losses.
    
#     Returns:
#     ------------------------------------------------------------------
#     - net_loss: `tf.Tensor`, The loss computed for the minibatch.
#     - aeloss: `tf.Tensor`, The reconstruction loss computed for the minibatch.
#     """

#     if aeloss_fun is not None:
        
#         aeloss_HVG = aeloss_fun(target, denoised_output['HVG_denoised'])
#         if LVG_target is not None:
#             aeloss_LVG = aeloss_fun(LVG_target, denoised_output['LVG_denoised'])
#             aeloss = 0.5*(aeloss_LVG + aeloss_HVG)
#         else:
#             aeloss = 1. * aeloss_HVG
#     else:
#         aeloss = 0.
    
#     net_loss = clust_weight * tf.reduce_mean(KLD(p, cluster_output_q)) + (2. - clust_weight) * aeloss
    
#     return net_loss, aeloss



# class TripletMarginLoss(nn.Module):
#     def __init__(self, margin, dist_type = 0):
#         super(TripletMarginLoss, self).__init__()
#         self.margin = margin
#         self.dist_type = dist_type
#     def forward(self, anchor, positive, negative):
#         #eucl distance
#         #dist = torch.sum( (anchor - positive) ** 2 - (anchor - negative) ** 2, dim=1)\
#         #        + self.margin
#         #ipdb.set_trace()
#         if(anchor.size()[0]==0):
#             return(torch.tensor(0.0, requires_grad=True))
#         else:
#             if self.dist_type == 0:
#                 dist_p = F.pairwise_distance(anchor ,positive)
#                 dist_n = F.pairwise_distance(anchor ,negative)
#             if self.dist_type == 1:
#                 dist_p = 1-F.cosine_similarity(anchor, positive)
#                 dist_n = 1-F.cosine_similarity(anchor, negative)
        
            
#             dist_hinge = torch.clamp(dist_p - dist_n + self.margin, min=0.0)
#             loss = torch.mean(dist_hinge)
    
#             return loss

