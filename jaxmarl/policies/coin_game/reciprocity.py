"""The reciprocity family: scripted Coin Game players that differ in how they answer a taken coin.

Each step a player picks a coin to go for, its own or the other player's, and takes a shortest
path to it on the 3x3 torus.  The pick is a softmax over the two coins:

    p(coin) is proportional to exp(weight[coin] - distance_weight * distance[coin])

with weight 0 on its own coin.  The weight on the other player's coin depends on whether the
player holds a grudge: ``w_nice`` without one, ``w_angry`` with one.  A player that does not
want the other's coin also avoids stepping on it on its way to its own.

The grudge is the player's only memory.  It counts the times the partner took the player's
coin (not counting a coin that lay on the same cell as the partner's own, which cannot be
left), and, for a forgiving type, goes down by one each time the player takes the partner's
coin while holding a grudge.  It lasts for the whole meta-episode (across inner episodes) and
starts at zero when a new type is drawn.

* cooperator:  never goes for the other's coin.
* defector:    goes for the nearer coin, whosever it is.
* tit_for_tat: cooperates; after each coin taken from it, takes one of the partner's.
* grim:        cooperates until the first coin is taken from it, then goes for the partner's
               coin for the rest of the meta-episode.

A cooperator and a tit-for-tat player act the same until their partner takes a coin, so only
a partner that defects can tell them apart; a grim player makes that test costly.

On top of that, every type moves uniformly at random with probability ``eps``.

The player reads its own observation, in any of the env's formats (flat or CNN, absolute or
egocentric), and whether the inner episode has just started.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from jaxmarl.environments.coin_game.coin_game import MOVES, CoinGame
from jaxmarl.policies.base import Policy, Population, Seen

NUM_ACTIONS = 5
STAY = 4
OWN, OTHER = 0, 1


@struct.dataclass
class ReciprocityParams:
    """A type of reciprocity player."""

    w_nice: jax.Array  # weight on the other's coin without a grudge
    w_angry: jax.Array  # weight on the other's coin with a grudge
    forgive: (
        jax.Array
    )  # 1: taking one of the partner's coins settles one grudge; 0: never
    eps: jax.Array  # probability of a uniformly random move


class ReciprocityMem(NamedTuple):
    grudge: jax.Array  # int32: coins taken from this player not yet answered
    coins: (
        jax.Array
    )  # int32[2, 2]: own and other coin, relative to this player, last step


class CoinView(NamedTuple):
    obs: jax.Array  # int[3, 3, 4]: [me, other player, my coin, other coin]
    fresh: jax.Array  # bool: the inner episode has just started


def _where(layer):
    """(row, col) of the single cell set in a [3, 3] layer."""
    flat = jnp.argmax(layer.reshape(-1))
    return jnp.stack([flat // 3, flat % 3]).astype(jnp.int32)


def _distance(rel):
    """Shortest-path length on the 3x3 torus to a cell at offset ``rel``."""
    return jnp.minimum(rel, 3 - rel).sum()


def _moves_towards(rel, avoid_rel, avoid):
    """Move probabilities [5] along a shortest path to offset ``rel``, split evenly.

    With ``avoid``, moves that land on ``avoid_rel`` are dropped unless no other shortest move
    is left.
    """
    landing = MOVES % 3  # the offset each move lands on
    shortest = jnp.array(
        [_distance((rel - landing[a]) % 3) < _distance(rel) for a in range(4)]
        + [_distance(rel) == 0]
    )
    hits = jnp.all(landing == avoid_rel, axis=-1)
    kept = shortest & ~hits
    use = jnp.where(avoid & kept.any(), kept, shortest)
    return use / use.sum()


class ReciprocityPolicy(Policy):
    def __init__(self, env, distance_weight: float = 1.0):
        if not isinstance(env, CoinGame):
            raise TypeError("ReciprocityPolicy is for CoinGame")
        self.distance_weight = distance_weight

    def view(self, env, state, obs, seat):
        return CoinView(obs=obs[seat].reshape(3, 3, 4), fresh=state.step == 0)

    def init(self, params):
        return ReciprocityMem(
            grudge=jnp.array(0, dtype=jnp.int32),
            coins=jnp.zeros((2, 2), dtype=jnp.int32),
        )

    def step(self, params: ReciprocityParams, mem: ReciprocityMem, seen: Seen):
        obs = seen.view.obs
        me = _where(obs[..., 0])
        other = (_where(obs[..., 1]) - me) % 3
        coins = jnp.stack([_where(obs[..., 2]), _where(obs[..., 3])])
        coins = (coins - me) % 3

        # what happened on the last move: last step's coins, relative to where I am now
        moved = seen.last_action >= 0
        last_coins = (mem.coins - MOVES[jnp.maximum(seen.last_action, 0)]) % 3
        valid = moved & ~seen.view.fresh
        # both coins on one cell go together, so that is not held against the partner
        shared = jnp.all(last_coins[OWN] == last_coins[OTHER])
        they_took_mine = valid & ~shared & jnp.all(other == last_coins[OWN])
        i_took_theirs = valid & jnp.all(last_coins[OTHER] == 0)
        settled = params.forgive * (i_took_theirs & (mem.grudge > 0))
        grudge = mem.grudge - settled + they_took_mine

        w_other = jnp.where(grudge > 0, params.w_angry, params.w_nice)
        scores = jnp.stack(
            [
                -self.distance_weight * _distance(coins[OWN]),
                w_other - self.distance_weight * _distance(coins[OTHER]),
            ]
        )
        pick = jax.nn.softmax(scores)
        intended = pick[OWN] * _moves_towards(
            coins[OWN], coins[OTHER], w_other < 0
        ) + pick[OTHER] * _moves_towards(coins[OTHER], coins[OWN], False)
        probs = (1.0 - params.eps) * intended + params.eps / NUM_ACTIONS

        new_mem = ReciprocityMem(grudge=grudge.astype(jnp.int32), coins=coins)
        return new_mem, jnp.log(probs)


RECIPROCITY_TYPES = {
    #               w_nice, w_angry, forgive
    "cooperator": (-8.0, -8.0, 1.0),
    "defector": (0.0, 0.0, 1.0),
    "tit_for_tat": (-8.0, 8.0, 1.0),
    "grim": (-8.0, 8.0, 0.0),
}


def make_population(env, eps: float = 0.05) -> Population:
    """The reciprocity types with a uniform prior."""
    names = tuple(RECIPROCITY_TYPES)
    table = jnp.array([RECIPROCITY_TYPES[name] for name in names])
    k = len(names)
    return Population(
        policy=ReciprocityPolicy(env),
        params=ReciprocityParams(
            w_nice=table[:, 0],
            w_angry=table[:, 1],
            forgive=table[:, 2].astype(jnp.int32),
            eps=jnp.full((k,), eps),
        ),
        probs=jnp.full((k,), 1.0 / k),
        names=names,
    )
