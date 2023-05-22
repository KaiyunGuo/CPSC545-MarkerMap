
import os
import contextlib
import queue
from functools import partial
import math
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import gc

import scipy.stats as stats

import anndata
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.ensemble import RandomForestClassifier

# from src.markermap.utils import split_data_into_dataloaders_no_test
# from utils import split_data_into_dataloaders_no_test
from src.myexp.markermap import BenchmarkableModel
from src.markermap.vae_models import train_save_model, train_model


EPSILON = 1e-40
MIN_TEMP = 0.0001


def trainAndGetRecon(hidden_layer_size, z_size, marker_indices, train_dataloader, val_dataloader, X_test, tmp_path, gpus):
    reconVAE = ReconstructionVAE(
                X_test.shape[1],
                hidden_layer_size,
                z_size,
                marker_indices
            )
    train_save_model(reconVAE, train_dataloader, val_dataloader, tmp_path, gpus=gpus, tpu_cores = None, 
                min_epochs = 25, max_epochs = 100, auto_lr = True, early_stopping_patience = 3,
            lr_explore_mode = 'linear', num_lr_rates=500)
    return reconVAE.get_reconstruction(X_test)




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


def make_encoder(input_size, hidden_layer_size, z_size, bias = True, batch_norm = True):
    """
    Construct encoder with 2 hidden layer used in VAE.
    args:
        input_size (int): Length of the input vector
        hidden_size (int): Size of the hidden layers
        z_size (int): size of encoded layer, latent size
        bias (bool): add a bias to the layers, defaults to True
        batch_norm (bool): use the batch norm in the layers, defaults to True
    returns: torch.nn.Sequential that encodes the input, the output layer for the mean, the output layer for the logvar
    """
    main_enc = nn.Sequential(
            *form_block(input_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = False)
            )

    enc_mean = nn.Linear(hidden_layer_size, z_size, bias = bias)
    enc_logvar = nn.Linear(hidden_layer_size, z_size, bias = bias)

    return main_enc, enc_mean, enc_logvar

def make_gaussian_decoder(output_size, hidden_size, z_size, bias = True, batch_norm = True):
    """
    Construct gaussian decoder with 1 hidden layer used in VAE. See Appendix C.2: https://arxiv.org/pdf/1312.6114.pdf
    args:
        output_size (int): Size of the reconstructed output of the VAE, probably the same as the input size
        hidden_size (int): Size of the hidden layer
        z_size (int): size of encoded layer, latent size
        bias (bool): add a bias to the layers, defaults to True
        batch_norm (bool): use the batch norm in the layers, defaults to True
    returns: torch.nn.Sequential that decodes the encoded representation
    """
    return nn.Sequential(
        *form_block(z_size, hidden_size, bias = bias, batch_norm = batch_norm),
        nn.Linear(hidden_size, output_size, bias = bias),
    )


class ReconstructionVAE(pl.LightningModule, BenchmarkableModel):
    """# Reconstruction with different markers
    train reconstruction on different markers and compare result
    """
    def __init__(
        self,
        input_size,
        hidden_layer_size,
        z_size,
        marker_indices,
        bias = True,
        batch_norm = True,
        lr = 0.000001,
        kl_beta = 0.1,
        decoder = None,
        dec_logvar = None,
    ):
        super(ReconstructionVAE, self).__init__()
        self.save_hyperparameters()

        output_size = input_size

        self.encoder, self.enc_mean, self.enc_logvar = make_encoder(input_size,
                hidden_layer_size, z_size, bias = bias, batch_norm = batch_norm)

        if (decoder is not None and dec_logvar is None) or (decoder is None and dec_logvar is not None):
            print(
                'VAE::__init__: WARNING! If decoder is specified, dec_logvar should also be specified, and vice versa'
            )

        if decoder is None:
            decoder = make_gaussian_decoder(
                output_size,
                hidden_layer_size,
                z_size,
                bias = bias,
                batch_norm = batch_norm,
            )

        if dec_logvar is None:
            dec_logvar = make_gaussian_decoder(
                output_size,
                hidden_layer_size,
                z_size,
                bias = bias,
                batch_norm = batch_norm,
            )

        self.decoder = decoder
        self.dec_logvar = dec_logvar

        self.lr = lr
        self.kl_beta = kl_beta
        self.batch_norm = batch_norm

        self.marker_indices = marker_indices

    def encode(self, x):
        marker = np.zeros_like(x)
        marker[:,self.marker_indices] = 1
        x = x * marker

        h1 = self.encoder(x)
        return self.enc_mean(h1), self.enc_logvar(h1)
    
    def decode(self, z):
        mu_x = self.decoder(z)  # reconstruction
        return mu_x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, x):
        mu_latent, logvar_latent = self.encode(x)
        z = self.reparameterize(mu_latent, logvar_latent)
        # mu_latent is used for the classifier and z for the VAE
        mu_x= self.decode(z)
        logvar_x = self.dec_logvar(z)

        return mu_x, logvar_x, mu_latent, logvar_latent

    def training_step(self, batch, batch_idx):
        x, y = batch

        mu_x, logvar_x, mu_latent, logvar_latent = self.forward(x)

        loss_recon = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent,
                                    kl_beta = self.kl_beta)

        if torch.isnan(loss_recon).any():
            raise Exception("nan loss during training")
        self.log('train_loss', loss_recon)
        return loss_recon

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            mu_x, logvar_x, mu_latent, logvar_latent = self(x)
            loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta)
        self.log('val_loss', loss)
        return loss

    def get_reconstruction(self, X):
        X = torch.Tensor(X)
        X.to(self.device)
        with torch.no_grad():
            mu_x = self.forward(X)[0].cpu().numpy()
        return mu_x


def loss_function_per_autoencoder(x, recon_x, logvar_x, mu_latent, logvar_latent, kl_beta = 0.1):
    # loss_rec = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # loss_rec = F.mse_loss(recon_x, x, reduction='sum')
    batch_size = x.size()[0]
    loss_rec = -torch.sum(
            (-0.5 * np.log(2.0 * np.pi))
            + (-0.5 * logvar_x)
            + ((-0.5 / torch.exp(logvar_x)) * (x - recon_x) ** 2.0)
            )

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp())
    loss = (loss_rec + kl_beta * KLD) / batch_size

    return loss


from torch.utils.data import DataLoader

def make_dataloaders(X, y, train_indices, val_indices, batch_size = 64, num_workers = 0, seed = None):
    """
    Split X and Y into training set (fraction train_size), validation set (fraction val_size)
    and the rest into a test set. train_size + val_size must be less than 1.
    Args:
        X (array): Input data
        y (vector): Output labels
        train_size (float): 0 to 1, fraction of data for train set
        val_size (float): 0 to 1, fraction of data for validation set
        batch_size (int): defaults to 64
        num_workers (int): number of cores for multi-threading, defaults to 0 for no multi-threading
        seed (int): defaults to none, set to reproduce experiments with same train/val split
    """
    if seed is not None:
        np.random.seed(seed)
    
    train_x = X[train_indices, :]
    val_x = X[val_indices, :]
    
    train_y = y[train_indices]
    val_y = y[val_indices]

    train_dataloader = get_dataloader(train_x, train_y, batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = get_dataloader(val_x, val_y, batch_size, shuffle=False, num_workers=num_workers)
    # test_dataloader = get_dataloader(test_x, test_y, batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader

def get_dataloader(X, y, batch_size, shuffle, num_workers=0):
    return DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X), torch.LongTensor(y)),
        batch_size=batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
    )


EMPTY_GROUP = -10000

def getTopVariances(arr, num_variances):
  assert num_variances <= len(arr)
  arr_copy = arr.copy()

  indices = set()
  for i in range(num_variances):
    idx = np.argmax(arr_copy)
    indices.add(idx)
    arr_copy[idx] = -1

  return indices

def l2(X1, X2):
  # (n x 1) matrix of l2 norm of each cell
  l2_cells = np.matmul(np.power(X1 - X2, 2), np.ones((X1.shape[1], 1)))
  return np.mean(np.power(l2_cells, 0.5))

def jaccard(X1, X2, num_variances):
  x_vars = np.var(X1, axis=0)
  recon_vars = np.var(X2, axis=0)

  x_top_vars = getTopVariances(x_vars, num_variances)
  recon_top_vars = getTopVariances(recon_vars, num_variances)

  return len(x_top_vars & recon_top_vars)/len(x_top_vars | recon_top_vars)

def analyzeVariance(X, recon_X, y, groups, num_variances):
  jaccard_index = EMPTY_GROUP*np.ones(len(groups))
  spearman_rho = EMPTY_GROUP*np.ones(len(groups))
  spearman_p = EMPTY_GROUP*np.ones(len(groups))

  for group in groups:
    group_indices = np.arange(len(y))[y == group]

    if (len(group_indices) == 0):
      continue

    X_group = X[group_indices, :]
    recon_X_group = recon_X[group_indices, :]

    jaccard_index[group] = jaccard(X_group, recon_X_group, num_variances)

    rho, p = stats.spearmanr(np.var(X_group, axis=0), np.var(recon_X_group, axis=0), alternative='greater')
    spearman_rho[group] = rho
    spearman_p[group] = p

  jaccard_overall = jaccard(X, recon_X, num_variances)
  spearman_rho_overall, spearman_p_overall = stats.spearmanr(
    np.var(X, axis=0),
    np.var(recon_X, axis=0),
    alternative='greater',
  )

  return jaccard_overall, jaccard_index, spearman_rho_overall, spearman_rho, spearman_p_overall, spearman_p

def getL2(X, recon_X, y, groups):
  l2_by_group = EMPTY_GROUP*np.ones(len(groups))
  for group in groups:
    group_indices = np.arange(len(y))[y == group]

    if (len(group_indices) == 0):
      continue

    X_group = X[group_indices, :]
    recon_X_group = recon_X[group_indices, :]

    l2_by_group[group] = l2(X_group, recon_X_group)

  l2_overall = l2(X, recon_X)
  return l2_overall, l2_by_group
