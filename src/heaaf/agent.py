"""Adaptive risk-evaluation layer: session-level MDP and a Double DQN agent.

Unlike a per-request classifier, HEAAF treats a clinical session as an episode.
Actions change the state that the next request is judged in (a step-up resets
the strong-authentication clock; a denial terminates the session), so the agent
learns *when* in a session to spend the user's attention, not merely whether a
single request looks anomalous.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (A, ACTIONS, AIDX, CHALLENGE_BYPASS_P, D, FEATURES, FIDX,
                     AgentConfig, RewardConfig)
from .nn import MLP


# ==========================================================================
# Environment
# ==========================================================================
class AccessEnv:
    """Replays a HEAAF-Bench trace as a set of episodic access sessions."""

    def __init__(self, df: pd.DataFrame, reward: RewardConfig,
                 seed: int = 0, shuffle: bool = True,
                 action_names: Optional[List[str]] = None,
                 clinical_costs: bool = True, bandit: bool = False):
        self.reward = reward
        self.action_names = list(action_names) if action_names else list(ACTIONS)
        self.clinical_costs = clinical_costs
        self.bandit = bandit
        self.rng = np.random.default_rng(seed)
        self.X = df[FEATURES].to_numpy(dtype=np.float64)
        self.y = df["y"].to_numpy(dtype=np.int64)
        self.emerg = df["emergency"].to_numpy(dtype=np.int64)
        self.sess = df["session"].to_numpy(dtype=np.int64)
        # index ranges per session
        bounds: Dict[int, List[int]] = {}
        for i, s in enumerate(self.sess):
            bounds.setdefault(s, [i, i])[1] = i
        self.sessions = list(bounds.keys())
        self.bounds = bounds
        self.shuffle = shuffle
        self._order: List[int] = []
        self._ptr = 0
        self.reset_all()

    # ------------------------------------------------------------------
    def reset_all(self) -> None:
        self._order = list(self.sessions)
        if self.shuffle:
            self.rng.shuffle(self._order)
        self._ptr = 0

    def reset(self) -> np.ndarray:
        if self.bandit:
            # single-step episodes: sweep the trace event by event
            self._ptr = (self._ptr + 1) % len(self.X)
            self.i = self.hi = self._ptr
            self.since_auth, self.fail = 0, 0.0
            return self._state()
        if self._ptr >= len(self._order):
            self.reset_all()
        s = self._order[self._ptr]
        self._ptr += 1
        lo, hi = self.bounds[s]
        self.i, self.hi = lo, hi
        self.since_auth = 0
        self.fail = 0.0
        return self._state()

    def _state(self) -> np.ndarray:
        x = self.X[self.i].copy()
        x[FIDX["auth_age"]] = min(1.0, self.since_auth / 10.0)
        x[FIDX["fail_recent"]] = min(1.0, self.fail)
        return x

    # ------------------------------------------------------------------
    def step(self, a: int) -> Tuple[np.ndarray, float, bool, Dict]:
        rc = self.reward
        act = self.action_names[a]
        x = self._state()
        y = int(self.y[self.i])
        emerg = int(self.emerg[self.i])
        sens = float(x[FIDX["res_sens"]])
        info = {"y": y, "action": act, "emergency": emerg, "bypassed": False}

        if y == 1:
            if act == "ALLOW":
                r = -(rc.breach_base + rc.breach_sens_gain * sens)
                done = self.i >= self.hi
            else:
                p = CHALLENGE_BYPASS_P[act]
                bypass = self.rng.random() < p
                info["bypassed"] = bool(bypass)
                if bypass:
                    r = -0.45 * (rc.breach_base + rc.breach_sens_gain * sens)
                    done = self.i >= self.hi
                else:
                    r = rc.correct_challenge[act]
                    done = True          # adversary is contained
            r *= rc.kappa
        else:
            fric = rc.friction_weight * rc.friction[act]
            if emerg and self.clinical_costs:
                fric += rc.emergency_penalty[act]
            r = (rc.allow_benign if act == "ALLOW" else 0.0) - fric
            if act == "DENY":
                done = True
            else:
                done = self.i >= self.hi

        # dynamics
        if act == "STEP_UP_STRONG":
            self.since_auth = 0
        elif act == "STEP_UP_OTP":
            self.since_auth = max(0, self.since_auth - 6)
        else:
            self.since_auth += 1
        if act.startswith("STEP_UP"):
            if y == 1 and info["bypassed"]:
                self.fail += 0.30
            elif y == 0 and self.rng.random() < 0.10:
                self.fail += 0.15
        self.fail *= 0.9

        if self.bandit:
            done = True
        if not done:
            self.i += 1
            nxt = self._state()
        else:
            nxt = np.zeros(D)
        return nxt, float(r), done, info


# ==========================================================================
# Replay buffer with class-balanced sampling (cf. RLAuth)
# ==========================================================================
class BalancedReplay:
    def __init__(self, capacity: int, dim: int, seed: int = 0):
        self.cap = capacity
        self.s = np.zeros((capacity, dim))
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity)
        self.s2 = np.zeros((capacity, dim))
        self.d = np.zeros(capacity)
        self.lab = np.zeros(capacity, dtype=np.int64)
        self.n = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, d, lab):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i], self.lab[i] = \
            s, a, r, s2, d, lab
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int, pos_frac: float):
        idx_all = np.arange(self.n)
        pos = idx_all[self.lab[:self.n] == 1]
        neg = idx_all[self.lab[:self.n] == 0]
        if len(pos) == 0 or len(neg) == 0:
            idx = self.rng.choice(idx_all, size=min(batch, self.n), replace=False)
        else:
            npos = int(batch * pos_frac)
            idx = np.concatenate([
                self.rng.choice(pos, size=npos, replace=len(pos) < npos),
                self.rng.choice(neg, size=batch - npos, replace=len(neg) < batch - npos)])
        return (self.s[idx], self.a[idx], self.r[idx],
                self.s2[idx], self.d[idx])


# ==========================================================================
# Agent
# ==========================================================================
@dataclass
class TrainLog:
    steps: List[int]
    ret: List[float]
    loss: List[float]


class DDQNAgent:
    def __init__(self, cfg: AgentConfig, n_actions: int = A, dim: int = D):
        self.cfg = cfg
        self.dim = dim
        self.n_actions = n_actions
        self.q = MLP([dim, *cfg.hidden, n_actions], seed=cfg.seed, lr=cfg.lr)
        self.tgt = self.q.clone()
        self.buf = BalancedReplay(cfg.buffer, dim, seed=cfg.seed + 7)
        self.rng = np.random.default_rng(cfg.seed + 13)
        self.step_count = 0

    # ------------------------------------------------------------------
    def eps(self) -> float:
        c = self.cfg
        frac = min(1.0, self.step_count / c.eps_decay_steps)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, s: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and self.rng.random() < self.eps():
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q(s)[0]))

    def q_values(self, X: np.ndarray) -> np.ndarray:
        return self.q(X)

    # ------------------------------------------------------------------
    def learn(self) -> float:
        c = self.cfg
        s, a, r, s2, d = self.buf.sample(c.batch, c.balanced_replay)
        q_next_online = self.q(s2)
        q_next_target = self.tgt(s2)
        if c.double_q:
            a_star = np.argmax(q_next_online, axis=1)
            boot = q_next_target[np.arange(len(a_star)), a_star]
        else:
            boot = q_next_target.max(axis=1)
        target = r + (1 - d) * c_gamma(self) * boot

        out, acts, pre = self.q.forward(s, cache=True)
        pred = out[np.arange(len(a)), a]
        err = pred - target
        err = np.clip(err, -10.0, 10.0)          # Huber-like stabilisation
        dY = np.zeros_like(out)
        dY[np.arange(len(a)), a] = err
        gW, gb = self.q.backward(acts, pre, dY, grad_clip=c.grad_clip)
        self.q.apply_grads(gW, gb)
        return float(np.mean(err ** 2))

    # ------------------------------------------------------------------
    def train(self, env: AccessEnv, gamma: float,
              log_every: int = 5000, verbose: bool = False) -> TrainLog:
        self.gamma = gamma
        c = self.cfg
        log = TrainLog([], [], [])
        s = env.reset()
        ep_ret, losses, rets = 0.0, [], []
        for t in range(c.episodes):
            self.step_count = t
            a = self.act(s)
            s2, r, done, info = env.step(a)
            self.buf.add(s, a, r, s2, float(done), info["y"])
            ep_ret += r
            s = env.reset() if done else s2
            if done:
                rets.append(ep_ret)
                ep_ret = 0.0
            if t >= c.warmup:
                losses.append(self.learn())
            if t % c.target_sync == 0:
                self.tgt.copy_from(self.q)
            if t % log_every == 0 and t > 0:
                log.steps.append(t)
                log.ret.append(float(np.mean(rets[-200:])) if rets else 0.0)
                log.loss.append(float(np.mean(losses[-500:])) if losses else 0.0)
                if verbose:
                    print(f"  step {t:>6}  eps={self.eps():.3f} "
                          f"return={log.ret[-1]:.2f} loss={log.loss[-1]:.3f}")
        return log


def c_gamma(agent: "DDQNAgent") -> float:
    return getattr(agent, "gamma", 0.9)


# ==========================================================================
# Risk head: turn action values into a calibrated probability
# ==========================================================================
class RiskHead:
    r"""Maps the action-value vector to a calibrated risk score in [0,1].

    The agent's advantage for *challenging* over *allowing*,

    .. math:: g(s) = \max_{a \neq \mathrm{ALLOW}} Q(s,a) - Q(s,\mathrm{ALLOW}),

    is monotone in the posterior probability that the request is malicious but
    is not itself a probability.  A one-dimensional logistic (Platt) map is
    fitted on the validation split so that the number the policy engine
    thresholds -- and that the explanation layer attributes -- is calibrated.
    """

    def __init__(self):
        self.w = 1.0
        self.b = 0.0
        self.mu = 0.0
        self.sd = 1.0

    @staticmethod
    def advantage(Q: np.ndarray) -> np.ndarray:
        allow = Q[:, AIDX["ALLOW"]]
        other = np.delete(Q, AIDX["ALLOW"], axis=1).max(axis=1)
        return other - allow

    def fit(self, Q: np.ndarray, y: np.ndarray, iters: int = 3000,
            lr: float = 0.5) -> "RiskHead":
        g = self.advantage(Q)
        self.mu, self.sd = float(g.mean()), float(g.std() + 1e-9)
        z = (g - self.mu) / self.sd
        w, b = 1.0, float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(w * z + b, -30, 30)))
            w -= lr * float(np.mean((p - y) * z))
            b -= lr * float(np.mean(p - y))
        self.w, self.b = float(w), float(b)
        return self

    def score_from_Q(self, Q: np.ndarray) -> np.ndarray:
        z = (self.advantage(Q) - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-np.clip(self.w * z + self.b, -30, 30)))

    def score(self, agent: DDQNAgent, X: np.ndarray) -> np.ndarray:
        return self.score_from_Q(agent.q_values(X))

    # -- differentiable surrogate used for attribution ------------------
    def logit_head_vector(self, Q_row: np.ndarray) -> np.ndarray:
        """Head vector h with h.Q = logit of the risk score at this point.

        The max over non-ALLOW actions is piecewise linear, so at any point the
        risk logit is an exact linear functional of Q with the selector below.
        """
        h = np.zeros(A)
        others = [i for i in range(A) if i != AIDX["ALLOW"]]
        j = others[int(np.argmax(Q_row[others]))]
        h[j] = self.w / self.sd
        h[AIDX["ALLOW"]] = -self.w / self.sd
        return h

    def risk_scalar(self, agent: DDQNAgent, X: np.ndarray) -> np.ndarray:
        return self.score(agent, X)
