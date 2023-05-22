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
from src.myexp.markermap import VAE_Gumbel, sample_subset, train_model


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



"""## VAE Gumbel topk"""
# idea of having a Non Instance Wise Gumbel that also has a state to keep consistency across batches
# probably some repetititon of code, but the issue is this class stuff, this is python 3 tho so it can be put into a good wrapper
# that doesn't duplicate code
class VAE_Gumbel_RunningState(VAE_Gumbel):
    # alpha is for the exponential average
    def __init__(
        self,
        input_size,
        hidden_layer_size,
        z_size,
        k,
        t = 0.01,
        temperature_decay = 0.99,
        method = 'mean',
        alpha = 0.9,
        bias = True,
        batch_norm = True,
        lr = 0.000001,
        kl_beta = 0.1,
        decoder = None,
        dec_logvar = None,
    ):
        super(VAE_Gumbel_RunningState, self).__init__(
            input_size,
            hidden_layer_size,
            z_size,
            k = k,
            t = t,
            temperature_decay = temperature_decay,
            bias = bias,
            batch_norm = batch_norm,
            lr = lr,
            kl_beta = kl_beta,
            decoder = decoder,
            dec_logvar = dec_logvar,
        )
        self.save_hyperparameters()
        self.method = method

        assert alpha < 1
        assert alpha > 0

        # flat prior for the features
        # need the view because of the way we encode
        self.register_buffer('logit_enc', torch.zeros(input_size).view(1, -1))

        self.alpha = alpha

    # training_phase determined by training_step
    def weights(self, x, training_phase=False):
        if training_phase:
            # FC + dropout + leakyReLU
            w = self.weight_creator(x)

            if self.method == 'mean':
                pre_enc = w.mean(dim = 0).view(1, -1)
            elif self.method == 'median':
                pre_enc = w.median(dim = 0)[0].view(1, -1)
            else:
                raise Exception("Invalid aggregation method inside batch of Non instancewise Gumbel")

            # self.alpha是论文中beta，alpha=0.9?
            self.logit_enc = (self.alpha) * self.logit_enc.detach() + (1-self.alpha) * pre_enc


            gumbel = training_phase
            subset_indices = sample_subset(self.logit_enc, self.k, self.t,
                            gumbel = gumbel, device = self.device)
            x = x * subset_indices
        else:
            mask = torch.zeros_like(x)
            mask.index_fill_(1, self.markers(), 1)
            x = x * mask

        return x

    def top_logits(self):
        with torch.no_grad():
            w = self.logit_enc.clone().view(-1)
            top_k_logits = torch.topk(w, k = self.k, sorted = True)[1]
            enc_top_logits = torch.nn.functional.one_hot(top_k_logits,
                            num_classes = self.hparams.input_size).sum(dim = 0)

            #subsets = sample_subset(w, model.k,model.t,True)
            subsets = sample_subset(w, self.k, self.t, gumbel = False, device = self.device)
            #max_idx = torch.argmax(subsets, 1, keepdim=True)
            #one_hot = Tensor(subsets.shape)
            #one_hot.zero_()
            #one_hot.scatter_(1, max_idx, 1)

        return enc_top_logits, subsets

    def markers(self):
        logits = self.top_logits()
        inds_running_state = torch.argsort(logits[0], descending = True)[:self.k]

        return inds_running_state

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        X,
        y,
        train_indices,
        val_indices,
        train_dataloader,
        val_dataloader,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            X (np.array): the full set of training data input X
            y (np.array): the full set of training data output y
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            train_dataloader (pytorch dataloader): dataloader for training data set
            val_dataloader (pytorch dataloader): dataloader for validation data set
            k (int): k value for the model, the number of markers to select
        """
        model = cls(**{**create_kwargs, 'k': k}) if k else cls(**create_kwargs)
        train_model(model, train_dataloader, val_dataloader, **train_kwargs)
        return model.markers().clone().cpu().detach().numpy()


# not doing multiple inheritance because GumbelClassifier is repeating itself
class PEDCCvae(VAE_Gumbel_RunningState):

    def __init__(
        self,
        input_size,
        hidden_layer_size,
        z_size,
        num_classes,
        k,
        c_path,
        t = 3.0,
        temperature_decay = 0.95,
        method = 'mean',
        alpha = 0.95,
        bias = True,
        batch_norm = True,
        lr = 0.000001,
        kl_beta = 0.1,
        decoder = None,
        dec_logvar = None,
        loss_tradeoff = 0
    ):

        super(PEDCCvae, self).__init__(
            input_size = input_size,
            hidden_layer_size = hidden_layer_size,
            z_size = z_size,
            k = k,
            t = t,
            temperature_decay = temperature_decay,
            method = method,
            alpha = alpha,
            batch_norm = batch_norm,
            bias = bias,
            lr = lr,
            kl_beta = kl_beta,
            decoder = decoder,
            dec_logvar = dec_logvar,
        )

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.register_buffer('loss_tradeoff', torch.as_tensor(1.0 * loss_tradeoff))
        self.map_dict = read_pkl(c_path)
        self.z_size = z_size

        if num_classes is None:
            self.classification_decoder = None
            self.classification_loss = None
        else:
            self.classification_decoder = nn.Sequential(
                    *form_block(z_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
                    nn.Linear(1*hidden_layer_size, num_classes, bias = bias),
                    nn.LogSoftmax(dim = 1)
                    )
            # self.classification_loss = nn.NLLLoss(reduction = 'sum')
            self.classification_loss = nn.MSELoss()


    def decode(self, z):
        mu_x = self.decoder(z)  # reconstruction
        return mu_x#, log_probs


    def forward(self, x, training_phase = False):
        x = self.weights(x, training_phase = training_phase)
        
        h1 = self.encoder(x)
        # en
        mu_latent = self.enc_mean(h1)
        logvar_latent = self.enc_logvar(h1)
        
        z = self.reparameterize(mu_latent, logvar_latent)
        # mu_latent is used for the classifier and z for the VAE
        # mu_x, log_probs = self.decode(mu_latent, z)
        mu_x = self.decode(z)

        logvar_x = self.dec_logvar(z)

        return mu_x, logvar_x, mu_latent, logvar_latent

    def training_step(self, batch, batch_idx):
        x, y = batch

        mu_x, logvar_x, mu_latent, logvar_latent = self.forward(x, training_phase = True)

        tensor_empty = torch.Tensor([])
        for label_index in y:
            tensor_empty = torch.cat((tensor_empty, self.map_dict[label_index.item()].float()), 0)
        label_tensor = tensor_empty.view(-1, self.z_size)

        # log_probs = self.forward(x, training_phase = True)
        # loss = self.loss_function(log_probs, y)

        loss_recon = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta)
        loss_classification = self.classification_loss(mu_latent, label_tensor)


        loss = loss_recon + loss_classification

        if torch.isnan(loss).any():
            raise Exception("nan loss during training")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            mu_x, logvar_x, mu_latent, logvar_latent = self.forward(x, training_phase = True)

            tensor_empty = torch.Tensor([])
            for label_index in y:
                tensor_empty = torch.cat((tensor_empty, self.map_dict[label_index.item()].float()), 0)
            label_tensor = tensor_empty.view(-1, self.z_size)

            # log_probs = self.forward(x, training_phase = True)
            # loss = self.loss_function(log_probs, y)

            loss_recon = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta)
            loss_classification = self.classification_loss(mu_latent, label_tensor)

            loss = loss_recon + loss_classification

        self.log('val_loss', loss)
        return loss


    # uses hard subsetting
    # returns log probs
    def predict_logprob(self, X):
        assert self.num_classes is not None
        log_probs = self.forward(X)[2]
        return log_probs

    # uses hard subsetting
    def predict_class(self, X):
        X = torch.Tensor(X)
        X.to(self.device)
        with torch.no_grad():
            log_probs = self.predict_logprob(X)
        return log_probs.max(dim=1)[1].cpu().numpy()

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
