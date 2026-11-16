"""Watch soup players: roll out an episode and save it as a GIF.

python -m jaxmarl.policies.overcooked_v2.demo --types runner plater --out soup.gif
"""

import argparse

import imageio
import jax
import numpy as np

from jaxmarl import make
from jaxmarl.policies import rollout, type_posterior
from jaxmarl.policies.overcooked_v2.common import MOVE_NAMES
from jaxmarl.policies.overcooked_v2.soup import SOUP_TYPES, make_population
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument(
        "--types", nargs=2, default=["runner", "plater"], choices=list(SOUP_TYPES)
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="soup.gif")
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()

    env = make("overcooked_v2", layout=args.layout, max_steps=args.steps)
    population = make_population(env)
    seats = {seat: population.only(name) for seat, name in zip(env.agents, args.types)}

    traj = jax.jit(lambda key: rollout(env, seats, key, args.steps))(
        jax.random.PRNGKey(args.seed)
    )

    frames = np.asarray(OvercookedV2Visualizer().render_sequence(traj.state))
    imageio.mimsave(args.out, frames, duration=1000.0 / args.fps, loop=0)

    deliveries = int(np.asarray(traj.reward[:, 0]).sum() // 20)
    print(
        f"{args.layout}, {args.steps} steps: {' + '.join(args.types)} delivered {deliveries} soups"
    )
    for i, seat in enumerate(env.agents):
        moves = np.bincount(np.asarray(traj.action[:, i]), minlength=len(MOVE_NAMES))
        log_post = type_posterior(population, env, traj, seat)[-1]
        belief = ", ".join(
            f"{n} {p:.2f}"
            for n, p in zip(population.names, np.exp(np.asarray(log_post)))
        )
        print(
            f"  {seat} ({args.types[i]}): moves {dict(zip(MOVE_NAMES, moves.tolist()))}"
        )
        print(f"    posterior over its type after the episode: {belief}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
