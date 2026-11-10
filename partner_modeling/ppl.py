"""A minimal probabilistic programming layer for discrete models.

A *program* is an ordinary Python function that calls ``sample``, ``flip``, ``observe``,
``condition`` or ``factor``.  The same program can be

* **enumerated** -- ``infer(program, *args)`` runs it once per path through its random choices
  and returns the exact distribution over its return value (the posterior, if it conditions);
* **simulated** -- ``simulate(program, *args, rng=...)`` runs it forward once, drawing each choice.

``infer`` calls nest: a program may call ``infer`` on another program and ``observe`` the
result, which is how a model of an agent that itself chooses stochastically is conditioned on
what that agent did.

Return values must be hashable.  Every random choice must have finite support.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_NEG_INF = float("-inf")


class Dist:
    """A finite discrete distribution over hashable values (zero-probability values are dropped).

    ``log_evidence`` is the log of the total weight before normalising: for a distribution
    returned by ``infer`` it is the log marginal likelihood of the program's conditions.
    """

    __slots__ = ("values", "probs", "logp", "log_evidence", "_index")

    def __init__(self, values: Sequence[Any], weights: Sequence[float]):
        w = np.asarray(weights, dtype=np.float64)
        keep = w > 0
        z = float(w[keep].sum())
        if not z > 0:
            raise ValueError("distribution has no mass (every execution was rejected)")
        self.values: Tuple[Any, ...] = tuple(v for v, k in zip(values, keep) if k)
        self.probs = w[keep] / z
        self.logp = np.log(self.probs)
        self.log_evidence = math.log(z)
        self._index: Optional[Dict[Any, int]] = None

    @classmethod
    def from_weights(cls, weights: Dict[Any, float]) -> "Dist":
        return cls(list(weights), list(weights.values()))

    @classmethod
    def uniform(cls, values: Iterable[Any]) -> "Dist":
        values = list(values)
        return cls(values, [1.0] * len(values))

    @classmethod
    def delta(cls, value: Any) -> "Dist":
        return cls([value], [1.0])

    def prob(self, value: Any) -> float:
        if self._index is None:
            self._index = {v: i for i, v in enumerate(self.values)}
        i = self._index.get(value)
        return float(self.probs[i]) if i is not None else 0.0

    def items(self):
        return zip(self.values, self.probs.tolist())

    def marginal(self, f: Callable[[Any], Any]) -> "Dist":
        """Distribution of ``f(value)``."""
        out: Dict[Any, float] = {}
        for v, p in self.items():
            k = f(v)
            out[k] = out.get(k, 0.0) + p
        return Dist.from_weights(out)

    def expect(self, f: Callable[[Any], float] = lambda v: v) -> float:
        return float(sum(p * f(v) for v, p in self.items()))

    def argmax(self) -> Any:
        return self.values[int(np.argmax(self.probs))]

    def draw(self, rng: np.random.Generator) -> Any:
        return self.values[int(np.searchsorted(np.cumsum(self.probs), rng.random(), side="right").clip(
            0, len(self.values) - 1))]

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return "Dist({" + ", ".join(f"{v!r}: {p:.3g}" for v, p in self.items()) + "})"


# ---------------------------------------------------------------------------
# The effects a program can perform
# ---------------------------------------------------------------------------

_STACK: List[Any] = []     # active handlers, innermost last


def sample(dist: Dist) -> Any:
    """A random choice."""
    return _STACK[-1].sample(dist)


def flip(p: float) -> bool:
    """A weighted coin.  Certain outcomes make no choice point."""
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return sample(Dist((True, False), (p, 1.0 - p)))


def factor(logw: float) -> None:
    """Multiply the weight of this execution by ``exp(logw)``."""
    _STACK[-1].factor(logw)


def observe(dist: Dist, value: Any) -> None:
    """Condition on a draw from ``dist`` having been ``value``."""
    p = dist.prob(value)
    factor(math.log(p) if p > 0 else _NEG_INF)


def condition(ok: bool) -> None:
    """Keep this execution only if ``ok``."""
    if not ok:
        factor(_NEG_INF)


class _Rejected(Exception):
    pass


class _Enumerate:
    """Follows ``prefix`` through the first choices, takes option 0 afterwards, and records the
    untaken options as new prefixes, so that every path is run exactly once."""

    def __init__(self, prefix: Tuple[int, ...]):
        self.prefix, self.path, self.logw, self.todo = prefix, [], 0.0, []

    def sample(self, dist: Dist) -> Any:
        i = len(self.path)
        if i < len(self.prefix):
            k = self.prefix[i]
        else:
            k, base = 0, tuple(self.path)
            self.todo.extend(base + (j,) for j in range(1, len(dist.values)))
        self.path.append(k)
        self.logw += dist.logp[k]
        return dist.values[k]

    def factor(self, logw: float) -> None:
        self.logw += logw
        if self.logw == _NEG_INF:
            raise _Rejected


class _Forward:
    """Draws every choice; accumulates the weight of ``factor`` / ``observe`` (likelihood weighting)."""

    def __init__(self, rng: np.random.Generator):
        self.rng, self.logw = rng, 0.0

    def sample(self, dist: Dist) -> Any:
        return dist.draw(self.rng)

    def factor(self, logw: float) -> None:
        self.logw += logw
        if self.logw == _NEG_INF:
            raise _Rejected


def _run(handler, program, args):
    _STACK.append(handler)
    try:
        return True, program(*args)
    except _Rejected:
        return False, None
    finally:
        _STACK.pop()


def infer(program: Callable, *args, method: str = "enumerate", n: int = 1000,
          rng: Optional[np.random.Generator] = None) -> Dist:
    """The distribution over ``program(*args)``'s return value, given its conditions.

    ``method="enumerate"`` is exact.  ``method="importance"`` is likelihood weighting with ``n``
    forward runs, for programs with too many paths to enumerate.
    """
    logw: Dict[Any, float] = {}         # log weights, so that large factors cannot overflow
    if method == "enumerate":
        todo: List[Tuple[int, ...]] = [()]
        while todo:
            h = _Enumerate(todo.pop())
            ok, value = _run(h, program, args)
            todo.extend(h.todo)
            if ok:
                logw[value] = np.logaddexp(logw.get(value, _NEG_INF), h.logw)
    elif method == "importance":
        rng = rng if rng is not None else np.random.default_rng()
        for _ in range(n):
            h = _Forward(rng)
            ok, value = _run(h, program, args)
            if ok:
                logw[value] = np.logaddexp(logw.get(value, _NEG_INF), h.logw - math.log(n))
    else:
        raise ValueError(f"unknown method {method!r}")
    if not logw:
        raise ValueError("distribution has no mass (every execution was rejected)")
    top = max(logw.values())
    d = Dist(list(logw), [math.exp(v - top) for v in logw.values()])
    d.log_evidence += top
    return d


def simulate(program: Callable, *args, rng: np.random.Generator) -> Any:
    """One forward run of ``program``.  Its conditions are ignored (a rejected run returns None)."""
    return _run(_Forward(rng), program, args)[1]
