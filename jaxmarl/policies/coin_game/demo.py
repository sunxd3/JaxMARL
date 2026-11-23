"""Watch reciprocity players: roll out a game and save it as a GIF.

python -m jaxmarl.policies.coin_game.demo --types defector tit_for_tat --out coins.gif
"""

import argparse

import imageio
import jax
import numpy as np

from jaxmarl import make
from jaxmarl.policies import rollout, type_posterior
from jaxmarl.policies.coin_game.reciprocity import RECIPROCITY_TYPES, make_population


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--types",
        nargs=2,
        default=["defector", "tit_for_tat"],
        choices=list(RECIPROCITY_TYPES),
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="coins.gif")
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()

    env = make("coin_game")
    population = make_population(env)
    seats = {seat: population.only(name) for seat, name in zip(env.agents, args.types)}

    traj = jax.jit(lambda key: rollout(env, seats, key, args.steps))(
        jax.random.PRNGKey(args.seed)
    )

    frames = [
        np.asarray(env.render(jax.tree.map(lambda x: x[t], traj.state)))
        for t in range(args.steps)
    ]
    imageio.mimsave(args.out, frames, duration=1000.0 / args.fps, loop=0)

    returns = np.asarray(traj.reward).sum(0)
    print(f"{args.steps} steps: {' vs '.join(args.types)}, returns {returns.tolist()}")
    for i, seat in enumerate(env.agents):
        log_post = type_posterior(population, env, traj, seat)[-1]
        belief = ", ".join(
            f"{n} {p:.2f}"
            for n, p in zip(population.names, np.exp(np.asarray(log_post)))
        )
        print(f"  {seat} ({args.types[i]}): posterior over its type: {belief}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
