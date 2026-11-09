"""Ideal observer for typed-intent partners: exact forward filter over (theta, intent).

Generative model (per step t, partner state s_t, shared memory m_t):

    theta_t ~ K(theta_t | theta_{t-1})                (optional switching kernel)
    i_t     ~ p_theta(i | s_t, m_t)                   (intent layer)
    a_t     ~ L_theta(a | ctrl_i(s_t))                (noise layer)
    m_{t+1} = u(s_t, m_t, a_t)                        (deterministic, theta-independent)

Because memory is theta-independent and replayable from (s, a), the filter over
theta is exact (no particle approximation) and the joint posterior over
(theta, i_t) is a per-step decomposition of the same quantity.

Diagnostics:
    * ``I_t = sum_theta b_t^-(theta) KL(pi_theta || pibar_t)`` -- expected information
      about theta carried by a_t under the current belief (mutual information).
    * per-step pairwise KL tensors for ``D_{theta theta'}``.
    * posterior traces, identification times, confusion matrices, calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from partner_modeling.partners.typed_intent import (
    NUM_ACTIONS, NUM_INTENTS, PartnerView, TypedIntentModel, _theta_tuple,
)
from partner_modeling.env_compat import Actions


@dataclass
class FilterResult:
    post: np.ndarray            # (T, n) b_t(theta) after a_t
    prior_pred: np.ndarray      # (T, n) b_t^-(theta) before a_t
    pibar: np.ndarray           # (T, 6) posterior predictive
    pi: np.ndarray              # (T, n, 6) pi_theta(.|s_t, m_t)
    loglik: np.ndarray          # (T, n) log pi_theta(a_t)
    info: np.ndarray            # (T,) I_t
    intent_post: np.ndarray     # (T, n, K) p(theta, i_t | h_t)
    intent_prior: np.ndarray    # (T, n, K) p_theta(i_t | s_t, m_t)
    mem: np.ndarray             # (T,) replayed memory before a_t
    decision: np.ndarray        # (T,) bool, >= 2 intents with differing controls valid
    adjacent: np.ndarray        # (T,) bool, other agent adjacent
    log_marginal: np.ndarray    # (T,) log p(a_t | h_{t-1})

    @property
    def T(self):
        return self.post.shape[0]

    def entropy_theta(self) -> np.ndarray:
        p = np.clip(self.post, 1e-300, 1)
        return -np.sum(self.post * np.log(p), axis=1)

    def entropy_intent_given_theta(self) -> np.ndarray:
        """E_theta[H(i_t | theta, h_t)] -- residual intent uncertainty."""
        joint = self.intent_post
        pth = joint.sum(-1, keepdims=True)
        cond = np.where(pth > 0, joint / np.maximum(pth, 1e-300), 0)
        h = -np.sum(np.where(cond > 0, cond * np.log(np.maximum(cond, 1e-300)), 0), axis=-1)
        return np.sum(pth[..., 0] * h, axis=-1)


def switching_kernel(n: int, hazard: float, prior: Optional[np.ndarray] = None) -> np.ndarray:
    """K[i, j] = P(theta_t = j | theta_{t-1} = i) = (1-h) [i==j] + h prior[j]."""
    prior = np.full(n, 1.0 / n) if prior is None else np.asarray(prior, float)
    return (1.0 - hazard) * np.eye(n) + hazard * prior[None, :]


class IdealObserver:
    def __init__(self, model: TypedIntentModel, thetas: Sequence, prior: Optional[np.ndarray] = None,
                 kernel: Optional[np.ndarray] = None):
        self.model = model
        self.thetas = list(thetas)
        self.n = len(self.thetas)
        self.prior = np.full(self.n, 1.0 / self.n) if prior is None else np.asarray(prior, float)
        self.kernel = kernel
        self.rho = np.array([_theta_tuple(t)[2] for t in self.thetas])

    def replay_memory(self, views: Sequence[PartnerView], actions: Sequence[int], mem0: int = -1) -> np.ndarray:
        mem = np.empty(len(actions), dtype=np.int64)
        m = mem0
        for t, (v, a) in enumerate(zip(views, actions)):
            mem[t] = m
            m = self.model.update_memory(v, m, int(a))
        return mem

    def run(self, views: Sequence[PartnerView], actions: Sequence[int], mem0: int = -1,
            reset_at: Optional[Sequence[int]] = None,
            switch_mask: Optional[Sequence[bool]] = None) -> FilterResult:
        """Filter one (or several concatenated) episode(s).

        ``reset_at``: step indices where memory is reset to -1 (episode boundaries);
        theta persists across them unless a switching kernel is given.
        ``switch_mask``: if given, the switching kernel is applied only at steps t with
        ``switch_mask[t]`` True (e.g. partner inventory transitions); otherwise every step.
        Memory is theta-independent, so a theta switch never desynchronises it.
        """
        T = len(actions)
        n, K = self.n, NUM_INTENTS
        eps = self.model.cfg.eps
        post = np.zeros((T, n))
        pp = np.zeros((T, n))
        pibar = np.zeros((T, NUM_ACTIONS))
        pis = np.zeros((T, n, NUM_ACTIONS))
        ll = np.zeros((T, n))
        info = np.zeros(T)
        ipost = np.zeros((T, n, K))
        iprior = np.zeros((T, n, K))
        mem = np.zeros(T, dtype=np.int64)
        dec = np.zeros(T, dtype=bool)
        adj = np.zeros(T, dtype=bool)
        lmarg = np.zeros(T)
        resets = set(reset_at or [])
        b = self.prior.copy()
        m = mem0
        for t in range(T):
            if t in resets:
                m = -1
            apply_k = self.kernel is not None and (switch_mask is None or bool(switch_mask[t]))
            bm = b @ self.kernel if apply_k else b
            A, P, f = self.model.dist_all(self.thetas, views[t], m)
            a = int(actions[t])
            mix = bm @ A
            # I_t = sum_theta bm KL(A_theta || mix)
            with np.errstate(divide="ignore", invalid="ignore"):
                kl = np.sum(A * (np.log(A) - np.log(mix)[None]), axis=1)
            info[t] = float(np.sum(bm * kl))
            # joint over (theta, intent) given a_t
            Lai = (1.0 - self.rho - eps)[:, None] * (f.ctrl[None, :] == a) \
                + self.rho[:, None] * (a == Actions.stay) + eps / NUM_ACTIONS
            joint = bm[:, None] * P * Lai
            z = joint.sum()
            ipost[t] = joint / z
            iprior[t] = P
            like = A[:, a]
            lmarg[t] = np.log(np.sum(bm * like))
            b = bm * like
            b = b / b.sum()
            post[t], pp[t], pibar[t], pis[t], ll[t] = b, bm, mix, A, np.log(like)
            mem[t], dec[t], adj[t] = m, f.decision_point, f.adjacent
            m = self.model.update_memory(views[t], m, a, f)
        return FilterResult(post=post, prior_pred=pp, pibar=pibar, pi=pis, loglik=ll, info=info,
                            intent_post=ipost, intent_prior=iprior, mem=mem, decision=dec,
                            adjacent=adj, log_marginal=lmarg)


# ---------------------------------------------------------------------------
# Diagnostics on filter results
# ---------------------------------------------------------------------------

def identification_time(post: np.ndarray, true_idx: int, thresh: float = 0.9) -> Dict[str, float]:
    """First step with b_t(true) >= thresh, and first step after which it stays >= thresh.
    Steps are 1-based counts of observed partner actions; inf when never reached."""
    hit = post[:, true_idx] >= thresh
    first = int(np.argmax(hit)) + 1 if np.any(hit) else np.inf
    if hit[-1]:
        not_hit = np.where(~hit)[0]
        stable = int(not_hit[-1]) + 2 if len(not_hit) else 1
    else:
        stable = np.inf
    return {"first": first, "stable": stable}


def pairwise_kl(pi: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """D[i, j] = mean_t KL(pi_i(.|s_t) || pi_j(.|s_t)) over steps (optionally masked)."""
    if mask is not None:
        pi = pi[mask]
    if len(pi) == 0:
        return np.full((pi.shape[1], pi.shape[1]), np.nan)
    lp = np.log(pi)
    # (T, n, 1, 6) vs (T, 1, n, 6)
    kl = np.sum(pi[:, :, None, :] * (lp[:, :, None, :] - lp[:, None, :, :]), axis=-1)
    return kl.mean(0)


def pairwise_kl_sum(pi: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is not None:
        pi = pi[mask]
    lp = np.log(pi)
    return np.sum(np.sum(pi[:, :, None, :] * (lp[:, :, None, :] - lp[:, None, :, :]), axis=-1), axis=0)


def randomized_pit(pi_true: np.ndarray, actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomised PIT for discrete predictions: Uniform(0,1) iff a_t ~ pi_true(.|s_t)."""
    cdf = np.cumsum(pi_true, axis=1)
    a = actions.astype(int)
    lo = np.where(a > 0, cdf[np.arange(len(a)), np.maximum(a - 1, 0)], 0.0)
    p = pi_true[np.arange(len(a)), a]
    return lo + rng.random(len(a)) * p
