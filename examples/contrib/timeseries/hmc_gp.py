import argparse

import numpy as np
import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
from pyro.contrib.timeseries import IndependentMaternGP
from pyro.ops.ssm_gp import MaternKernel
from pyro.nn.module import PyroModule, PyroSample

from gp_models import download_data


pyro.enable_validation(__debug__)


class RandomMaternKernel(MaternKernel, PyroModule):
    pass


class RandomGP(IndependentMaternGP, PyroModule):
    Kernel = RandomMaternKernel
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prior_dist = dist.Normal(0.0, 2 * torch.ones(self.obs_dim)).to_event(1)
        self.kernel.log_length_scale = PyroSample(prior_dist)
        self.kernel.log_kernel_scale = PyroSample(prior_dist)
        self.kernel.log_obs_noise_scale = PyroSample(prior_dist)



def main(args):
    download_data()
    data = np.loadtxt('eeg.dat', delimiter=',', skiprows=19)
    print("[raw data shape] {}".format(data.shape))
    data = torch.tensor(data[::20, :-1]).double()
    data = data[:100, :1]  # choose a single output dimension
    print("[data shape after thinning] {}".format(data.shape))

    T, obs_dim = data.shape

    # standardize data
    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    torch.manual_seed(args.seed)

    gp = RandomGP(nu=1.5, obs_dim=obs_dim).double()

    def model():
        pyro.factor("y", gp.log_prob(data).sum(-1))

    nuts_kernel = NUTS(model, max_tree_depth=4)
    mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=100, num_chains=1)
    mcmc.run()
    mcmc.summary(prob=0.5)


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.5.1')
    parser = argparse.ArgumentParser(description="contrib.timeseries hmc example usage")
    parser.add_argument("-s", "--seed", default=0, type=int)
    args = parser.parse_args()

    main(args)
