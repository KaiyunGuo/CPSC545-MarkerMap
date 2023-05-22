import math
import logging
import os
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.ensemble import RandomForestClassifier



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


class GumbelFCClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_layer_size, z_size, num_classes, k, batch_norm = True, t = 2, temperature_decay = 0.9, method = 'mean', alpha = 0.99, bias = True, lr = 0.000001,
            min_temp = MIN_TEMP):
        super(GumbelFCClassifier, self).__init__()
        self.save_hyperparameters()
        assert temperature_decay > 0
        assert temperature_decay <= 1


        self.encoder = nn.Sequential(
            *form_block(input_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            nn.Linear(1*hidden_layer_size, num_classes, bias = bias),
            nn.LogSoftmax(dim = 1)
        )

        # self.weight_creator = nn.Sequential(
        #     *form_block(input_size, hidden_layer_size, batch_norm = batch_norm),
        #     nn.Dropout(),
        #     *form_block(hidden_layer_size, hidden_layer_size, batch_norm = batch_norm),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_layer_size, input_size)
        # )
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

        assert alpha < 1
        assert alpha > 0

        # flat prior for the features
        # need the view because of the way we encode
        self.register_buffer('logit_enc', torch.zeros(input_size).view(1, -1))

        self.alpha = alpha
        self.loss_function = nn.NLLLoss()

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
        else:
            mask = torch.zeros_like(x)
            mask.index_fill_(1, self.markers(), 1)

            x = x * mask

        h1 = self.encoder(x)
        # en
        return h1

    def forward(self, x, training_phase = False):
        log_probs = self.encode(x, training_phase = training_phase)

        return log_probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.forward(x, training_phase = True)
        loss = self.loss_function(log_probs, y)
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
        with torch.no_grad():
            log_probs = self.forward(x, training_phase = False)
            loss = self.loss_function(log_probs, y)
            acc = (y == log_probs.max(dim=1)[1]).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
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



# from torch.utils.data import DataLoader

# def make_dataloaders(X, y, train_indices, val_indices, batch_size = 64, num_workers = 0, seed = None):
#     """
#     Split X and Y into training set (fraction train_size), validation set (fraction val_size)
#     and the rest into a test set. train_size + val_size must be less than 1.
#     Args:
#         X (array): Input data
#         y (vector): Output labels
#         train_size (float): 0 to 1, fraction of data for train set
#         val_size (float): 0 to 1, fraction of data for validation set
#         batch_size (int): defaults to 64
#         num_workers (int): number of cores for multi-threading, defaults to 0 for no multi-threading
#         seed (int): defaults to none, set to reproduce experiments with same train/val split
#     """
#     if seed is not None:
#         np.random.seed(seed)
    
#     train_x = X[train_indices, :]
#     val_x = X[val_indices, :]
    
#     train_y = y[train_indices]
#     val_y = y[val_indices]

#     train_dataloader = get_dataloader(train_x, train_y, batch_size, shuffle=True, num_workers=num_workers)
#     val_dataloader = get_dataloader(val_x, val_y, batch_size, shuffle=False, num_workers=num_workers)
#     # test_dataloader = get_dataloader(test_x, test_y, batch_size, shuffle=False, num_workers=num_workers)

#     return train_dataloader, val_dataloader

# def get_dataloader(X, y, batch_size, shuffle, num_workers=0):
#     return DataLoader(
#         torch.utils.data.TensorDataset(torch.Tensor(X), torch.LongTensor(y)),
#         batch_size=batch_size,
#         shuffle = shuffle,
#         num_workers = num_workers,
#     )

class GumbelFCVarClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_layer_size, z_size, num_classes, var_X, k, batch_norm = True, t = 2, temperature_decay = 0.9, method = 'mean', alpha = 0.99, bias = True, lr = 0.000001,
            min_temp = MIN_TEMP):
        super(GumbelFCVarClassifier, self).__init__()
        self.save_hyperparameters()
        assert temperature_decay > 0
        assert temperature_decay <= 1


        self.encoder = nn.Sequential(
            *form_block(input_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            nn.Linear(1*hidden_layer_size, num_classes, bias = bias),
            nn.LogSoftmax(dim = 1)
        )

        # self.weight_creator = nn.Sequential(
        #     *form_block(input_size, hidden_layer_size, batch_norm = batch_norm),
        #     nn.Dropout(),
        #     *form_block(hidden_layer_size, hidden_layer_size, batch_norm = batch_norm),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_layer_size, input_size)
        # )
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
        self.var_X = var_X

        assert alpha < 1
        assert alpha > 0

        # flat prior for the features
        # need the view because of the way we encode
        self.register_buffer('logit_enc', torch.zeros(input_size).view(1, -1))

        self.alpha = alpha
        self.loss_function = nn.NLLLoss()

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
        else:
            mask = torch.zeros_like(x)
            mask.index_fill_(1, self.markers(), 1)

            x = x * mask

        h1 = self.encoder(x)
        # en
        return h1

    def forward(self, x, training_phase = False):
        log_probs = self.encode(x, training_phase = training_phase)

        return log_probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.forward(x, training_phase = True)
        loss1 = self.loss_function(log_probs, y)

        marker_var = np.sum(self.var_X[self.markers().clone().cpu().detach().numpy()])
        loss2 = 1 - (marker_var / np.sum(self.var_X)) + np.sum(marker_var<=0) / self.k
  
        loss = loss1 + loss2

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
        with torch.no_grad():
            log_probs = self.forward(x, training_phase = False)
            loss1 = self.loss_function(log_probs, y)

            marker_var = np.sum(self.var_X[self.markers().clone().cpu().detach().numpy()])
            loss2 = 1 - (marker_var/np.sum(self.var_X)) + np.sum(marker_var<=0) / self.k

            loss = loss1 + loss2

            acc = (y == log_probs.max(dim=1)[1]).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
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