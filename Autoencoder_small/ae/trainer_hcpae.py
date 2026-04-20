import numpy as np
import torch
import torch.nn.functional as F
from ae.distributions import rand_cirlce2d
import sys
sys.path.append("../src")
import base



def _hcp_distance(encoded_samples, distribution_samples, p=2, device='cpu'):

    embedding_dim = distribution_samples.size(1)
    A = encoded_samples.detach().cpu().numpy() #convert to Numpy array
    A1 = base.hilbert_order(A)

    B = distribution_samples.detach().cpu().numpy() #convert to Numpy array
    B1 = base.hilbert_order(B)

    hcp_distance = encoded_samples[A1,:] - distribution_samples[B1,:]
    hcp_distance = torch.pow(hcp_distance, p)
    
    return hcp_distance.mean()*embedding_dim


def HCP_distance(encoded_samples, distribution_fn=rand_cirlce2d, p=2, device='cpu'):

    batch_size = encoded_samples.size(0)
    z = distribution_fn(batch_size).to(device)
    hpd = _hcp_distance(encoded_samples, z, p, device)   #_new_python1
    return hpd


class HCPAEBatchTrainer:

    def __init__(self, autoencoder, optimizer, distribution_fn, p=2, weight=5, device=None):
        self.model_ = autoencoder
        self.optimizer = optimizer
        self._distribution_fn = distribution_fn
        self.embedding_dim_ = self.model_.encoder.embedding_dim_
        self.p_ = p
        self.weight = weight
        self._device = device if device else torch.device('cpu')

    def __call__(self, x):
        return self.eval_on_batch(x)

    def train_on_batch(self, x):
        self.optimizer.zero_grad()
        evals = self.eval_on_batch(x)
        evals['loss'].backward()
        self.optimizer.step()
        return evals

    def test_on_batch(self, x):
        self.optimizer.zero_grad()
        evals = self.eval_on_batch(x)
        return evals

    def eval_on_batch(self, x):
        x = x.to(self._device)
        recon_x, z = self.model_(x)
        bce = F.binary_cross_entropy(recon_x, x)
        l1 = F.l1_loss(recon_x, x)

        _swd = HCP_distance(z, self._distribution_fn, self.p_, self._device)
        w2 = float(self.weight) * _swd  # approximate wasserstein-2 distance
        loss =  w2 +bce + l1
        print(f"bce={bce.item():.4f}, l1={l1.item():.4f}, "
              f"HCP={_swd.item():.4f}, w2={w2.item():.4f}")

        return {
            'loss': loss,
            'bce': bce,
            'l1': l1,
            'w2': w2,
            'encode': z,
            'decode': recon_x
        }
