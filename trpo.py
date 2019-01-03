import torch
import numpy as np
from distributions import DiagonalGaussian
from torch.autograd import Variable
from helpers import get_flat_params, set_flat_params, get_flat_grads, compute_advantage_returns, normal_log_density


def compute_surr_losses(policy, trajs):
    observations = np.asarray(trajs["state"])
    actions = trajs["actions"]

    old_means = trajs["dist"]["means"]
    old_std = trajs["dist"]["log_std"]
    advantage = trajs["advantages"]

    _, new_dists = policy.actions(observations)
    new_means = new_dists["mean"]
    new_std = new_dists["log_std"]

    dist_old = DiagonalGaussian(old_means, old_std)
    dist_new = DiagonalGaussian(new_means, new_std)
    #print(actions.size())

    importance_sampling = dist_new.logli_ratio(dist_old, actions)
    surr_loss = -(importance_sampling * advantage).mean()
    return surr_loss


def compute_kl_div(policy, trajs, subsample_ratio=1.0):
    observations = np.asarray(trajs["state"])
    old_means = trajs["dist"]["means"]
    old_std = trajs["dist"]["log_std"]

    ep_len = observations.shape[0]
    if subsample_ratio < 1.0:
        mask = np.zeros(ep_len, dtype=np.int32)
        mask_ids = np.random.choice(ep_len, size=int(ep_len * subsample_ratio), replace=False)
        mask[mask_ids] = 1 
        subsample_obs = observations[mask]
        subsample_dists = dict(means=old_means[mask], log_std=old_std[mask])
    else:
        subsample_obs = observations
        subsample_dists = dict(means=old_means, log_std=old_std)


    _, new_dists = policy.actions(subsample_obs)
    new_means = new_dists["mean"]
    new_std = new_dists["log_std"]
    old_means = subsample_dists["means"]
    old_std = subsample_dists["log_std"]
    dist_old = DiagonalGaussian(old_means, old_std)
    dist_new = DiagonalGaussian(new_means, new_std)
    kl_divs = dist_old.kl_div(dist_new).mean()
    return kl_divs

def fvp(policy, f_kl, grads, v, eps=1e-5, damping=1e-8):

    flat_params = get_flat_params(policy.network.parameters())
    set_flat_params(policy.network, (flat_params + eps * v))
    policy.clear_grads()
    kl_loss = f_kl()
    print("KLTHAT MATTERS ",kl_loss)
    kl_loss.backward(retain_graph=True)
    grads_e = get_flat_grads(policy.network)
    set_flat_params(policy.network, flat_params)
    finite_diff = (grads_e - grads)/eps + damping * flat_params
    return finite_diff

def fvp_Hess(policy, f_kl, v, damping=1e-1):
    policy.clear_grads()
    kl_loss = f_kl()
    kl_loss.backward(create_graph=True)
    flat_grads_kl = get_flat_grads(policy.network)
    kl_sum = (flat_grads_kl * v).sum()
    grads_grads = torch.autograd.grad(kl_sum, policy.network.parameters(), retain_graph=True)
    flat_grads2_kl = torch.cat([grads.contiguous().view(-1) for grads in grads_grads])
    return flat_grads2_kl + v * damping


def conjugate_gradient(Fx, b, cg_iters=10, residual_tol=1e-10):

    p = b.clone()
    r = b.clone()
    x = torch.zeros(b.size())
    rr = torch.dot(r,r)
    for i in range(cg_iters):
        Avp = Fx(p)
        alpha = rr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rr = torch.dot(r,r)
        beta = new_rr / rr
        p = r + beta * p
        rr = new_rr
        if rr < residual_tol:
            break
    return x


def linesearch(f, x0, dx, expected_improvement, y0=None, backtrack_ratio=0.8, max_backtracks=15, accept_ratio=0.1, tol=1e-7):
    #print("IMP=",expected_improvement)
    if expected_improvement >= tol:
        if y0 is None:
            y0 = f(x0)

        for ratio in backtrack_ratio**np.arange(max_backtracks):
            x = x0 - ratio * dx
            with torch.no_grad():
                y = f(x)
            actual_improvement = y0 - y
            if (actual_improvement / (expected_improvement * ratio)) >= accept_ratio:
                print("ExpectedImprovement: {}".format(expected_improvement*ratio))
                print("ActualImprovement: {}".format(actual_improvement))
                print("ImprovementRatio: {}".format(actual_improvement / (expected_improvement * ratio)))
                return x

    return x0




def trpo_step(policy, baseline, trajs, step_size=0.01, use_linesearch=True, subsample_ratio=1.0, gamma=0.99, gae_lambda=0.97):
    ###function in function helper function####
    compute_advantage_returns(trajs, baseline, gamma, gae_lambda)
    #policy.network.train()
    def f_loss():
        return compute_surr_losses(policy, trajs)

    def f_kl():
        return compute_kl_div(policy, trajs, subsample_ratio)

    policy.clear_grads()

    surr_loss = f_loss()
    surr_loss.backward(retain_graph=True)
    flat_grad = get_flat_grads(policy.network)
    #print("SANITY CHECK:{}".format(torch.dot(flat_grad, flat_grad)))
    policy.clear_grads()

    kl_loss = f_kl()
    #print(kl_loss)
    kl_loss.backward(retain_graph=True)
    flat_grad_kl = get_flat_grads(policy.network)

    #

    def Fx(v):
        return fvp(policy, f_kl, flat_grad_kl, v)

    def FH(v):
        return fvp_Hess(policy, f_kl, v)
    
    dir_cg = conjugate_gradient(Fx, flat_grad)
    #print("DIRN I=",torch.dot(dir_cg,Fx(dir_cg)))
    scale = torch.sqrt(2.0 * step_size / (torch.dot(dir_cg,Fx(dir_cg)) + 1e-8))
    print("SCALE:", scale)

    descent_step = dir_cg * scale

    curr_flat_params = get_flat_params(policy.network.parameters())
    #new_flat_params = None
    if use_linesearch:
        expected_improvement = torch.dot(flat_grad, descent_step)
        #print(expected_improvement)

        def f_barrier(x):
            set_flat_params(policy.network, x)
            with torch.no_grad():
                surr_loss = f_loss()
                kl = f_kl()
            return surr_loss.data + 1e100 * max((kl.data - step_size), 0.)

        new_flat_params = linesearch(
            f_barrier,
            x0=curr_flat_params,
            dx=descent_step,
            y0=surr_loss.data.clone(),
            expected_improvement=expected_improvement
        )

    else:
        new_flat_params = curr_flat_params - descent_step

    set_flat_params(policy.network, new_flat_params)