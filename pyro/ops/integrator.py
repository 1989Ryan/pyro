# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from copy import deepcopy
from cProfile import run
from torch.autograd import grad
import pyro.poutine as poutine
import torch

def run_prog(
    model,
    z,
    transforms,
    *args,
    **kwargs,
):
    """
    run probabilistic program to get new `z`
    given the current z: the value of the each step
    need to construct the trace from the value z
    """
    conditioned_model = poutine.condition(model, data=z)
    trace = poutine.trace(conditioned_model).get_trace(*args, **kwargs)
    # new_trace = poutine.trace(poutine.replay(model, trace=trace)).get_trace()
    new_trace = dict(trace.nodes)       
    new_z = {site_name: new_trace[site_name]["value"] for site_name in new_trace \
        if site_name not in ["_INPUT", "_RETURN", "obs"]}
    is_cont = {site_name: new_trace[site_name]["is_cont"] for site_name in new_trace \
            if site_name not in ["_INPUT", "_RETURN", "obs"]}
    is_cont_vector = torch.tensor([new_trace[site_name]["is_cont"] for site_name in new_trace\
            if site_name not in ["_INPUT", "_RETURN", "obs"]])
    return new_z, is_cont, is_cont_vector



def velocity_verlet(
    z, r, potential_fn, kinetic_grad, step_size, num_steps=1, z_grads=None
):
    r"""
    Second order symplectic integrator that uses the velocity verlet algorithm.

    :param dict z: dictionary of sample site names and their current values
        (type :class:`~torch.Tensor`).
    :param dict r: dictionary of sample site names and corresponding momenta
        (type :class:`~torch.Tensor`).
    :param callable potential_fn: function that returns potential energy given z
        for each sample site. The negative gradient of the function with respect
        to ``z`` determines the rate of change of the corresponding sites'
        momenta ``r``.
    :param callable kinetic_grad: a function calculating gradient of kinetic energy
        w.r.t. momentum variable.
    :param float step_size: step size for each time step iteration.
    :param int num_steps: number of discrete time steps over which to integrate.
    :param torch.Tensor z_grads: optional gradients of potential energy at current ``z``.
    :return tuple (z_next, r_next, z_grads, potential_energy): next position and momenta,
        together with the potential energy and its gradient w.r.t. ``z_next``.
    """
    z_next = z.copy()
    r_next = r.copy()
    for _ in range(num_steps):
        z_next, r_next, z_grads, potential_energy = _single_step_verlet(
            z_next, r_next, potential_fn, kinetic_grad, step_size, z_grads
        )
    return z_next, r_next, z_grads, potential_energy


def _single_step_verlet(z, r, potential_fn, kinetic_grad, step_size, z_grads=None):
    r"""
    Single step velocity verlet that modifies the `z`, `r` dicts in place.
    """

    z_grads = potential_grad(potential_fn, z)[0] if z_grads is None else z_grads
    # print(r)
    # print(z)
    for site_name in r:
        r[site_name] = r[site_name] + 0.5 * step_size * (
            -z_grads[site_name]
        )  # r(n+1/2)

    r_grads = kinetic_grad(r)
    for site_name in z:
        z[site_name] = z[site_name] + step_size * r_grads[site_name]  # z(n+1)

    z_grads, potential_energy = potential_grad(potential_fn, z)
    for site_name in r:
        r[site_name] = r[site_name] + 0.5 * step_size * (-z_grads[site_name])  # r(n+1)

    return z, r, z_grads, potential_energy

def leapfrog_discontiouous(
    z, r, is_cont, model, transforms, potential_fn, kinetic_grad, step_size, num_steps=1, z_grads=None
):
    r"""
    Leapfrog algorithm for discontinuous HMC
    """
    z_next = z.copy()
    r_next = r.copy()
    z_0 = z.copy()
    r_0 = r.copy()
    for _ in range(num_steps):
        z_next, r_next, z_grads, potential_energy, r_0 = _single_step_leapfrog_discontiuous(
            z_next, r_next, z_0, r_0, is_cont, transforms, model, potential_fn, kinetic_grad, step_size, z_grads
        )
    return z_next, r_next, z_grads, potential_energy, r_0


def _single_step_leapfrog_discontiuous(z, r, z_0, r_0, is_cont, transforms, model, potential_fn, kinetic_grad, step_size, z_grads=None):
    r"""
    Single step leapfrog algorithm that modifies the  `z` and `r` dicts in place by Laplace momentum
    for discontinuous HMC
    """
    # update the momentum 
    z_grads = potential_grad(potential_fn, z)[0] if z_grads is None else z_grads
    for site_name in z_grads:
        r[site_name] = r[site_name] + 0.5 * step_size * (
            -z_grads[site_name]
        ) * is_cont[site_name]  # r(n+1/2)

    # update the variable
    r_grads = kinetic_grad(r)
    for site_name in z:
        z[site_name] = z[site_name] + 0.5 * step_size * r[site_name] * is_cont[site_name]  # z(n+1)

    z, is_cont, is_cont_vector = run_prog(model, z, transforms)
    # conduct coordinate integration TODO: discontinuous flag
    # TODO: permutation
    disc_indices = torch.flatten(torch.nonzero(~is_cont_vector, as_tuple=False))
    perm = torch.randperm(len(disc_indices))
    disc_indices_permuted = disc_indices[perm]
    for j in disc_indices_permuted:
        if j >= len(z):
            continue 
        z, r, is_cont, is_cont_vector, r_0 = _coord_integrator(z, r, z_0, r_0, int(j.item()), model, transforms, potential_fn, kinetic_grad, step_size, z_grads)

    # update the variable
    z_grads, potential_energy = potential_grad(potential_fn, z)
    for site_name in z:
        z[site_name] = z[site_name] + 0.5 * step_size * r[site_name] * is_cont[site_name]  # r(n+1)
    
    z, is_cont, is_cont_vector = run_prog(model, z, transforms)

    # update momentum
    z_grads, potential_energy = potential_grad(potential_fn, z)
    for site_name in z_grads:
        r[site_name] = r[site_name] + 0.5 * step_size * (
            -z_grads[site_name]
        ) * is_cont[site_name]  # r(n+1/2)

    return z, r, z_grads, potential_energy, r_0


def _coord_integrator(z, r, z_0, r_0, idx, model, transforms, potential_fn, kinetic_grad, step_size, z_grads=None):
    r"""
    Coordinatewise integrator for dynamics with Laplace momentum for discontinuous HMC
    """
    U = potential_fn(z)
    new_z = deepcopy(z)
    site_name = list(new_z.keys())[idx]
    new_z[site_name] += step_size * torch.sign(r[site_name])
    new_z, new_is_cont, new_is_cont_vec = run_prog(model, z, transforms)
    new_U = potential_fn(new_z)
    delta_U = new_U - U
    if not math.isfinite(new_U) or torch.abs(r[site_name]) <= delta_U:
        r[site_name] = -r[site_name]
    else:
        r[site_name] -= torch.sign(r[site_name]) * delta_U
        N2 = len(new_z)
        N = len(z)
        site_name_list = list(new_z.keys())
        if N2 > N:
            # extend everything to the higher dimension
            gauss = torch.distributions.Normal(0, 1).sample([N2-N])
            laplace = torch.distributions.Laplace(0, 1).sample([N2-N])
            r_padding = gauss * new_is_cont_vec[N:N2] + laplace * ~new_is_cont_vec
            for i in range(N, N2):
                site_name = site_name_list[i]
                r[site_name] = r_padding[i-N]
                r_0[site_name] = r_padding[i-N]
        else:
            # truncate everything to the lower dimension
            for i in range(N2, N):
                site_name = site_name_list[i]
                r.pop(site_name)
                r_0.pop(site_name)
        assert len(new_z) == len(r)
        assert len(r_0) == len(r)
    return new_z, r, new_is_cont, new_is_cont_vec, r_0

def velocity_verlet_with_extension(
    z, r, potential_fn, kinetic_grad, num_steps=1, z_grads=None
):
    r"""
    Second order symplectic integrator that uses the velocity verlet algorithm with dimenstion
    extension
    """
    pass

def _single_step_verlet_with_extension(z, r, potential_fn, kinetic_grad, step_size, z_grads=None):
    r"""
    Single step velocity verlet taht modifies the `z`, `r` dicts in place with dimension extension
    TODO: add extension function with checking whether the `z` is at the domain of the question
    """
 
    z_grads = potential_grad(potential_fn, z)[0] if z_grads is None else z_grads
    # print(r)
    # print(z)
    for site_name in r:
        r[site_name] = r[site_name] + 0.5 * step_size * (
            -z_grads[site_name]
        )  # r(n+1/2)
    
    r_grads = kinetic_grad(r)
    for site_name in z:
        z[site_name] = z[site_name] + step_size * r_grads[site_name]  # z(n+1)

    z_grads, potential_energy = potential_grad(potential_fn, z)
    for site_name in r:
        r[site_name] = r[site_name] + 0.5 * step_size * (-z_grads[site_name])  # r(n+1)

    return z, r, z_grads, potential_energy

def _extention(z):
    """
    TODO: extend the dimensionality of `z` and `r`, used in np-hmc
    """
    pass

def potential_grad(potential_fn, z):
    """
    Gradient of `potential_fn` w.r.t. parameters z.

    :param potential_fn: python callable that takes in a dictionary of parameters
        and returns the potential energy.
    :param dict z: dictionary of parameter values keyed by site name.
    :return: tuple of `(z_grads, potential_energy)`, where `z_grads` is a dictionary
        with the same keys as `z` containing gradients and potential_energy is a
        torch scalar.
    """
    z_keys, z_nodes = zip(*z.items())
    for node in z_nodes:
        node.requires_grad_(True)
    try:
        potential_energy = potential_fn(z)
    # deal with singular matrices
    except RuntimeError as e:
        if "singular U" in str(e) or "input is not positive-definite" in str(e):
            grads = {k: v.new_zeros(v.shape) for k, v in z.items()}
            return grads, z_nodes[0].new_tensor(float("nan"))
        else:
            raise e

    grads = grad(potential_energy, z_nodes, allow_unused=True)
    for node in z_nodes:
        node.requires_grad_(False)
    grad_ret = dict(zip(z_keys, grads))
    if grad_ret["start"] is None:
        grad_ret["start"] = torch.tensor(0.0)
    assert len(grad_ret) == len(z)
    return grad_ret, potential_energy.detach()
