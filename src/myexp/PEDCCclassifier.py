import pickle
import math
import logging
import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.ensemble import RandomForestClassifier

# from config import latent_variable_dim,PEDCC_ui,model_path,epoches


def read_pkl(c_path):
    f = open(c_path,'rb')
    a = pickle.load(f)
    f.close()
    return a


EPSILON = 1e-40
MIN_TEMP = 0.0001


def form_block(in_size, out_size, batch_norm = True, bias = True):
    """
    Constructs a fully connected layer with bias, batch norm, and then leaky relu activation function
    args:
        in_size (int): layer input size
        out_size (int): layer output size
        batch_norm (bool): use the batch norm in the layers, defaults to True
        bias (bool): add a bias to the layers, defaults to True
    returns (array): the layers specified
    """
    layers = []
    layers.append(nn.Linear(in_size, out_size, bias = bias))
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_size))
    layers.append(nn.LeakyReLU())
    return layers


class pedccClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_layer_size, z_size, num_classes, k,
                c_path, batch_norm = True, t = 2, temperature_decay = 0.9, method = 'mean', alpha = 0.99, bias = True, lr = 0.000001,
            min_temp = MIN_TEMP):
        super(pedccClassifier, self).__init__()
        self.save_hyperparameters()
        assert temperature_decay > 0
        assert temperature_decay <= 1

        self.small_layer_size = hidden_layer_size/2


        self.encoder = nn.Sequential(
            *form_block(input_size, self.small_layer_size , bias = bias, batch_norm = batch_norm),
            *form_block(self.small_layer_size , self.small_layer_size , bias = bias, batch_norm = False),
            nn.Linear(self.small_layer_size , z_size, bias = True),
            nn.LeakyReLU()
        )


        self.decoder = main_dec = nn.Sequential(
            *form_block(z_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            nn.Linear(1*hidden_layer_size, num_classes, bias = bias),
            nn.LogSoftmax(dim = 1)
        )

        self.weight_creator = nn.Sequential(
            *form_block(input_size, hidden_layer_size, batch_norm = batch_norm),
            nn.Dropout(),
            *form_block(hidden_layer_size, hidden_layer_size, batch_norm = batch_norm),
            nn.Linear(hidden_layer_size, input_size)
        )

        self.method = method
        self.k = k
        self.batch_norm = batch_norm
        self.register_buffer('t', torch.as_tensor(1.0 * t))
        self.min_temp = min_temp
        self.temperature_decay = temperature_decay
        self.lr = lr
        self.bias = bias
        self.num_classes = num_classes
        self.map_dict = read_pkl(c_path)
        self.z_size = z_size

        assert alpha < 1
        assert alpha > 0

        # flat prior for the features
        # need the view because of the way we encode
        self.register_buffer('logit_enc', torch.zeros(input_size).view(1, -1))

        self.alpha = alpha
        # self.loss_function = nn.NLLLoss()
        self.loss_function = nn.MSELoss()
    
    # def encode2(self, x):


    # training_phase determined by training_step
    def encode(self, x, training_phase=False):
        if training_phase:
            w = self.weight_creator(x)

            if self.method == 'mean':
                pre_enc = w.mean(dim = 0).view(1, -1)
            elif self.method == 'median':
                pre_enc = w.median(dim = 0)[0].view(1, -1)
            else:
                raise Exception("Invalid aggregation method inside batch of Non instancewise Gumbel")

            self.logit_enc = (self.alpha) * self.logit_enc.detach() + (1-self.alpha) * pre_enc

            gumbel = training_phase
            subset_indices = sample_subset(self.logit_enc, self.k, self.t, gumbel = gumbel, device = self.device)

            x = x * subset_indices
            # x = x[:, subset_indices]
        else:
            mask = torch.zeros_like(x)
            mask.index_fill_(1, self.markers(), 1)

            x = x * mask

        h1 = self.encoder(x)
        # en
        return h1

    def forward(self, x, training_phase = False):
        h = self.encode(x, training_phase = training_phase)


        # log_probs = self.decoder(h)
        # return log_probs
        return h


    def training_step(self, batch, batch_idx):
        x, y = batch

        tensor_empty = torch.Tensor([])
        for label_index in y:
            tensor_empty = torch.cat((tensor_empty, self.map_dict[label_index.item()].float()), 0)
        label_tensor = tensor_empty.view(-1, self.z_size)

        # log_probs = self.forward(x, training_phase = True)
        # loss = self.loss_function(log_probs, y)
        output_classifier = self.forward(x)
        loss = self.loss_function(output_classifier, label_tensor)


        if torch.isnan(loss).any():
            raise Exception("nan loss during training")
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.t = max(torch.as_tensor(self.min_temp, device = self.device), self.t * self.temperature_decay)


        loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log("epoch_avg_train_loss", loss)
        return None

    def validation_step(self, batch, batch_idx):
        x, y = batch

        tensor_empty = torch.Tensor([])
        for label_index in y:
            tensor_empty = torch.cat((tensor_empty, self.map_dict[label_index.item()].float()), 0)
        label_tensor = tensor_empty.view(-1, self.z_size)

        with torch.no_grad():
            output_classifier = self.forward(x)
            loss = self.loss_function(output_classifier, label_tensor)
            # log_probs = self.forward(x, training_phase = False)
            # loss = self.loss_function(log_probs, y)
            # acc = (y == log_probs.max(dim=1)[1]).float().mean()
        self.log('val_loss', loss)
        # self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def top_logits(self):
        with torch.no_grad():
            w = self.logit_enc.clone().view(-1)
            top_k_logits = torch.topk(w, k = self.k, sorted = True)[1]
            enc_top_logits = torch.nn.functional.one_hot(top_k_logits, num_classes = self.hparams.input_size).sum(dim = 0)

            #subsets = sample_subset(w, model.k,model.t,True)
            subsets = sample_subset(w, self.k, self.t, device = self.device, gumbel = False)
            #max_idx = torch.argmax(subsets, 1, keepdim=True)
            #one_hot = Tensor(subsets.shape)
            #one_hot.zero_()
            #one_hot.scatter_(1, max_idx, 1)

        return enc_top_logits, subsets

    def markers(self):
        logits = self.top_logits()
        inds_running_state = torch.argsort(logits[0], descending = True)[:self.k]

        return inds_running_state

# 全部从Reparameterizable Subset Sampling via Continuous Relaxations复制的
def gumbel_keys(w, EPSILON):
    """
    Sample some gumbels, adapted from
    https://github.com/ermongroup/subsets/blob/master/subsets/sample_subsets.py
    Args:
        w (Tensor): Weights for each element, interpreted as log probabilities
        epsilon (float): min difference for float equalities
    """
    uniform = (1.0 - EPSILON) * torch.rand_like(w) + EPSILON
    z = -torch.log(-torch.log(uniform))
    w = w + z
    return w


def continuous_topk(w, k, t, device, separate=False, EPSILON = EPSILON):
    """
    Continuous relaxation of discrete variables, equations 3, 4, and 5
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
        separate (bool): defaults to false, swap to true for debugging
        epsilon (float): min difference for float equalities
    """

    # https://github.com/ermongroup/subsets/blob/master/subsets/sample_subsets.py
    if separate:
        khot_list = []
        onehot_approx = torch.zeros_like(w, dtype = torch.float32, device = device)
        for i in range(k):
            max_mask = (1 - onehot_approx) < EPSILON
            khot_mask = 1 - onehot_approx
            khot_mask[max_mask] = EPSILON
            w = w + torch.log(khot_mask)
            onehot_approx = F.softmax(w/t, dim = -1)
            khot_list.append(onehot_approx)
        return torch.stack(khot_list)
    # https://github.com/ermongroup/subsets/blob/master/subsets/knn/sorting_operator.py
    else:
        relaxed_k = torch.zeros_like(w, dtype = torch.float32, device = device)
        onehot_approx = torch.zeros_like(w, dtype = torch.float32, device = device)
        for i in range(k):
            max_mask = (1 - onehot_approx) < EPSILON
            khot_mask = 1 - onehot_approx
            khot_mask[max_mask] = EPSILON
            w = w + torch.log(khot_mask)
            onehot_approx = F.softmax(w/t, dim = -1)
            relaxed_k = relaxed_k + onehot_approx
        return relaxed_k


def sample_subset(w, k, t, device, separate = False, gumbel = True, EPSILON = EPSILON):
    """
    Sample k elements using the continuous relaxation of discrete variables.
    A good default value of t is 0.0001, but let VAE gumbel constructor decide that.
    Adapted from: https://github.com/ermongroup/subsets/blob/master/subsets/sample_subsets.py
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
        separate (bool): defaults to false, swap to true for debugging
    """
    assert EPSILON > 0
    if gumbel:
        w = gumbel_keys(w, EPSILON)
    return continuous_topk(w, k, t, device, separate = separate, EPSILON = EPSILON)