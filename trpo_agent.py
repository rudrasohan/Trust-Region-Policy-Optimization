import torch

import torch
import numpy as np
from copy import deepcopy
from distributions import DiagonalGaussian
from torch.autograd import Variable
from helpers import get_flat_params, set_flat_params, get_flat_grads, compute_advantage_returns, sample_trajectories
#from simplepg.simple_utils import test_once
from distributions import DiagonalGaussian

class TRPO(object):

    def __init__(self,
                 env,
                 policy,
                 baseline,
                 step_size=0.01, 
                 use_linesearch=True,
                 subsample_ratio=1.0,
                 gamma=0.995,
                 gae_lambda=0.97,
                 damping=1e-1,
                 cg_iters=10,
                 residual_tol=1e-10,
                 ent_coeff=0.0000
                 ):
        
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.step_size = step_size
        self.use_linesearch = use_linesearch
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.damping = damping
        self.cg_iters = cg_iters
        self.residual_tol = residual_tol
        self.ent_coeff = ent_coeff
        self.trajs = None

    def compute_surr_losses(self, x):
        observations = np.asarray(self.trajs["state"])
        actions = self.trajs["actions"]
        model = deepcopy(self.policy)
        set_flat_params(model.network, x)
        dist_new = model.get_dists(observations)

        advantage = self.trajs["advantages"].data

        dist_old = self.policy.get_dists(observations)

        importance_sampling = dist_new.logli_ratio(dist_old, actions).data
        surr_loss = -(importance_sampling * advantage).mean()
        return surr_loss

    def compute_kl_div(self, policy):
        observations = np.asarray(self.trajs["state"])
        dist_new = policy.get_dists(observations)
        dist_new.detach()
        dist_old = self.policy.get_dists(observations)
        kl_divs = dist_old.kl_div(dist_new).mean()
        return kl_divs

    def hvp(self, v):
        self.policy.clear_grads()
        kl_1 = self.compute_kl_div(self.policy)
        grads = torch.autograd.grad(kl_1, self.policy.network.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        kl_v = torch.dot(flat_grad_kl, v)
        grad_grads = torch.autograd.grad(kl_v, self.policy.network.parameters())
        fvp = torch.cat([grad.contiguous().view(-1) for grad in grad_grads]).data
        return fvp + self.damping * v.data

    def conjugate_gradient(self, b):

        p = b.clone().data
        r = b.clone().data
        x = torch.zeros(b.size())
        rr = torch.dot(r,r)
        for i in range(self.cg_iters):
            #print("CG_ITERS",i)
            Avp = self.hvp(p)
            alpha = rr / torch.dot(p, Avp)
            x += alpha * p
            r -= alpha * Avp
            new_rr = torch.dot(r,r)
            beta = new_rr / rr
            p = r + beta * p
            rr = new_rr
            if rr < self.residual_tol:
                break
        return x

    def linesearch(self, x0, dx, expected_improvement_rate, fval=None, backtrack_ratio=0.5, max_backtracks=10, accept_ratio=0.1):
        fval = self.compute_surr_losses(x0)
        print("fval before", fval.item())
        for ratio in backtrack_ratio**np.arange(max_backtracks):
            x = x0.data + ratio * dx.data
            new_fval = self.compute_surr_losses(x)
            actual_improvement = fval - new_fval
            expected_improvement = expected_improvement_rate * ratio
            actual_ratio = actual_improvement / expected_improvement
            print("a/e/r", actual_improvement.item(), expected_improvement.item(), actual_ratio.item())

            if actual_ratio.item() > accept_ratio and actual_improvement.item() > 0:
                print("fval after", new_fval.item())
                return True, x
        return False, x0

    def step(self):
        paths = sample_trajectories(self.env, self.policy, num_samples=5000)
        sumi = 0.0
        avg = []
        for num in range(len(paths["rewards"])):
            if paths["done"][num]:
                sumi += paths["rewards"][num]
            else:
                avg.append(sumi)
                sumi = 0.0
        #print(avg)
        #print("Avg Reward = {}".format(np.mean(avg)))
        #print("Max Reward = {}".format(np.max(avg)))
        #print("Min Reward = {}".format(np.min(avg)))
        compute_advantage_returns(paths, self.baseline, self.gamma, self.gae_lambda)
        self.trajs = paths

        new_prob = self.trajs['act_prob']
        old_prob = new_prob.detach()
        advantage = self.trajs["advantages"].data
        importance_sampling = (new_prob - old_prob).exp()
        act_prob = new_prob
        entropy = (act_prob * act_prob.exp()).mean()
        surr_loss = -(importance_sampling * advantage).mean() - self.ent_coeff * entropy
        print("Surrogate Loss",surr_loss)
        print("Entropy",entropy)
        self.policy.clear_grads()

        surr_loss.backward(retain_graph=True)
        flat_grads = get_flat_grads(self.policy.network)

        cg_dir = self.conjugate_gradient(-flat_grads)
        shs = 0.5 * torch.dot(cg_dir, self.hvp(cg_dir))
        lm = torch.sqrt(shs / self.step_size)
        print("Lagrange Multiplier",lm)
        descent_dir = cg_dir / lm
        gdotstep = -torch.dot(flat_grads, cg_dir)

        curr_flat_params = get_flat_params(self.policy.network) 
        status, params = self.linesearch(x0=curr_flat_params, dx=descent_dir, expected_improvement_rate=(gdotstep / lm))
        self.baseline.update(self.trajs)
        old = deepcopy(self.policy)
        old.network.load_state_dict(self.policy.network.state_dict())
        if any(np.isnan(params.data.numpy())):
            print("Skipping update...")
        else:
            print(status)
            if status:
                print("UPDATING PARAMS")
                set_flat_params(self.policy.network, params)

        kl_old_new = self.compute_kl_div(old)
        print("KL Div", kl_old_new)
        self.trajs = None
        del paths