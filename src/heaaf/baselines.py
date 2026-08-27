"""Baseline access-control policies used for comparison.

Each baseline is a stand-in for a family of systems in the literature:

===========  ====================================================================
B1 AlwaysMFA maximal-assurance zero trust: challenge every request
B2 Static    login-time-only authentication (the legacy hospital baseline)
B3 RBA       multiplicative statistical risk-based authentication in the style
             of Freeman et al., as deployed and measured by Wiefling et al.
B4 XAI-Static supervised detector plus post-hoc explanations on a fixed data
             set -- the design used by recent explainable-authentication work
B5 RLAuth-style deep RL anomaly detector: adaptive but opaque, binary output
===========  ====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config import A, ACTIONS, AIDX, D, FEATURES


class BasePolicy:
    name = "base"

    def risk(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def act(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def explains(self) -> bool:
        return False


class AlwaysMFA(BasePolicy):
    name = "B1-AlwaysMFA"

    def risk(self, X):
        return np.ones(len(X))

    def act(self, X):
        return np.full(len(X), AIDX["STEP_UP_OTP"], dtype=np.int64)


class StaticPassword(BasePolicy):
    name = "B2-StaticLogin"

    def risk(self, X):
        return np.zeros(len(X))

    def act(self, X):
        return np.full(len(X), AIDX["ALLOW"], dtype=np.int64)


class FreemanRBA(BasePolicy):
    r"""Multiplicative risk-based authentication.

    Risk is the product over features of a smoothed likelihood ratio
    :math:`p(x_i \mid \text{attack}) / p(x_i \mid \text{legitimate})`,
    estimated by histogram binning on the training split, exactly as in the
    classical RBA formulation.  Two thresholds map the score onto the same
    action set as HEAAF so the comparison is like for like.
    """

    name = "B3-RBA"

    def __init__(self, bins: int = 12, smooth: float = 1.0):
        self.bins = bins
        self.smooth = smooth

    def fit(self, X, y):
        self.edges, self.lr = [], []
        for j in range(X.shape[1]):
            e = np.quantile(X[:, j], np.linspace(0, 1, self.bins + 1))
            e = np.unique(e)
            if len(e) < 3:
                e = np.linspace(X[:, j].min(), X[:, j].max() + 1e-6, 4)
            self.edges.append(e)
            idx = np.clip(np.digitize(X[:, j], e[1:-1]), 0, len(e) - 2)
            nb = len(e) - 1
            pa = np.bincount(idx[y == 1], minlength=nb) + self.smooth
            pl = np.bincount(idx[y == 0], minlength=nb) + self.smooth
            self.lr.append(np.log((pa / pa.sum()) / (pl / pl.sum())))
        return self

    def _score(self, X):
        s = np.zeros(len(X))
        for j in range(X.shape[1]):
            e = self.edges[j]
            idx = np.clip(np.digitize(X[:, j], e[1:-1]), 0, len(e) - 2)
            s += self.lr[j][idx]
        return s

    def calibrate(self, X, y):
        s = self._score(X)
        self.mu, self.sd = float(s.mean()), float(s.std() + 1e-9)
        z = (s - self.mu) / self.sd
        w, b = 1.0, float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
        for _ in range(3000):
            p = 1 / (1 + np.exp(-np.clip(w * z + b, -30, 30)))
            w -= 0.5 * float(np.mean((p - y) * z))
            b -= 0.5 * float(np.mean(p - y))
        self.w, self.b = w, b
        return self

    def risk(self, X):
        z = (self._score(X) - self.mu) / self.sd
        return 1 / (1 + np.exp(-np.clip(self.w * z + self.b, -30, 30)))

    def act(self, X, tau_low=0.25, tau_high=0.60):
        r = self.risk(X)
        a = np.full(len(r), AIDX["ALLOW"], dtype=np.int64)
        mid = 0.5 * (tau_low + tau_high)
        a[(r >= tau_low) & (r < mid)] = AIDX["STEP_UP_OTP"]
        a[(r >= mid) & (r < tau_high)] = AIDX["STEP_UP_STRONG"]
        a[r >= tau_high] = AIDX["DENY"]
        return a


class XAIStatic(BasePolicy):
    """Supervised random forest with post-hoc explanations (offline only)."""

    name = "B4-XAI-Static"

    def __init__(self, seed: int = 0, n_estimators: int = 200):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=None, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=1, random_state=seed)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def risk(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def act(self, X, tau=0.5):
        r = self.risk(X)
        a = np.full(len(r), AIDX["ALLOW"], dtype=np.int64)
        a[r >= tau] = AIDX["DENY"]
        return a

    def explains(self) -> bool:
        return True   # but only off-line, after the decision


class RLAuthStyle(BasePolicy):
    """Binary, bandit-formulated deep RL detector without a policy engine."""

    name = "B5-RLAuth-style"

    def __init__(self, agent, head):
        self.agent = agent
        self.head = head

    def risk(self, X):
        Q = self.agent.q_values(X)
        adv = Q[:, 1] - Q[:, 0]
        z = (adv - self.head["mu"]) / self.head["sd"]
        return 1 / (1 + np.exp(-np.clip(self.head["w"] * z + self.head["b"], -30, 30)))

    def act(self, X, tau=0.5):
        r = self.risk(X)
        a = np.full(len(r), AIDX["ALLOW"], dtype=np.int64)
        a[r >= tau] = AIDX["DENY"]
        return a
