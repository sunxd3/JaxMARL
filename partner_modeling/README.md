# partner_modeling

Scripted Overcooked-V2 partners whose action probabilities are known exactly, and an exact
Bayesian ideal observer over their types. The purpose is to study how a focal agent models its
partner and decides, against a known ground truth.

## The partner (typed-intent)

Each step the partner (agent_1):

1. **picks a goal** (intent) by a softmax over 9 intents (deliver, pick up soup, get plate, add
   ingredient, fetch from pile, fetch from counter, stage on counter, yield, idle). The scores are
   type-weighted values minus 0.5 × path length, plus a bonus for keeping its last intent;
2. **moves toward it** with a deterministic controller, the same for every type;
3. **adds noise**: it stands still with probability ρ, and acts uniformly at random 3% of the time.

`dist()` returns the exact action distribution and `sample()` draws from it. Memory is only
the last intent, and it can be replayed from the observed history, so the ideal observer is
exact.

| family | types | what varies |
|---|---|---|
| `typed_intent` | 12 | role runner / plater / flex (fetch vs plate weights 6/1.5, 1.5/6, 4/4) × staging pot-direct / counter-stager × ρ 0.05 / 0.25 |
| `typed_intent_x` | 8 | exclusive roles: runner_x never fetches plates, plater_x never fetches ingredients × staging × ρ |
| `typed_intent_g` | 8 | graded: each step, with probability `excl_q`, the off-role intents are masked (q = 0 is soft, q = 1 is exclusive) |

## Files

| file | contents |
|---|---|
| `partners/typed_intent.py` | `TypedIntentTheta` (types and grids), `TypedIntentConfig`, `TypedIntentModel` (`dist`, `sample`, `update_memory`, `features`), `PartnerView` |
| `partners/grid_nav.py` | `GridNav`: layout masks, all-pairs shortest paths (BFS) and pot-state decoding that the controllers are built on |
| `observer.py` | `IdealObserver`: exact filter over (type, intent), with an optional switching kernel; identification time, KL tools |
| `rollouts.py` | `make_study_env`, `run_episode`, focal agents (static, random, scripted coop / fetcher / plater / shuttle), the strategy library used for best responses, role-lock variants, trajectory rendering |
| `teammates.py` | `make_teammate`, `HeuristicTeammateSpec`, `make_theta_set`, `sample_theta` |
| `checks.py` | validation suite: sampler vs `dist` (χ²), independence across episodes, focal view coverage |
| `sweep.py` | parallel runner: focal × type × seed episodes with the ideal observer, saved as `results.pkl` |
| `env_compat.py` | the Overcooked-V2 names the code uses |

## Setup and tests

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r partner_modeling/requirements-cpu.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest tests/partner_modeling -q      # 32 tests, about 1 min
```

## Example

```python
import jax
from partner_modeling.observer import IdealObserver, identification_time
from partner_modeling.partners.typed_intent import TypedIntentConfig, TypedIntentModel, TypedIntentTheta
from partner_modeling.rollouts import env_layout, make_study_env, run_episode, scripted_strategies
from partner_modeling.teammates import HeuristicTeammateSpec, make_teammate

env = make_study_env("coord_ring", max_steps=100)
layout = env_layout(env)
thetas = TypedIntentTheta.grid()                       # the 12 soft types
partner = make_teammate(HeuristicTeammateSpec(family="typed_intent", theta=thetas[6].preset_name()),
                        env, "agent_1")
focal = scripted_strategies(layout)["coop"]
log = run_episode(env, partner, focal, jax.random.PRNGKey(0), max_steps=100)

observer = IdealObserver(TypedIntentModel(layout, agent_id=1, config=TypedIntentConfig()), thetas)
res = observer.run(log.views, log.partner_actions)    # res.post[t] = posterior over the 12 types
print(res.post[-1, 6], identification_time(res.post, 6))
```

## Environment

`make_study_env` builds JaxMARL's `OvercookedV2` with the study settings (`STUDY_SETTINGS` in
`rollouts.py`):
- view radius 2;
- random agent positions;
- negative rewards;
- a new recipe sampled on each delivery;
- no random reset.

`full_obs=True` sets the view radius to the whole grid and changes nothing else.

The partner, the scripted focal agents and the observer read the full env state, so the view
radius only affects the `obs` arrays.

## Equivalence with the original study code

Checked on 2026-11-09 against the code the study's results came from, with the same JAX 0.5.3
stack on both sides.

- **Episode comparison:** 1,024 100-step episodes, run with the same seeds, are **bit-identical**
  in every step's state, both agents' actions, the partner's memory and the rewards.
  - Each of the 1,024 episodes is distinct, and 537 score.
  - Coverage:
    - `typed_intent` × 12 types on coord_ring, counter_circuit, cramped_room, forced_coord and
      asymm_advantages;
    - `typed_intent_x` and `typed_intent_g` (q = 0.5) × 8 types on coord_ring and counter_circuit;
    - the `"shortest"` controller × 12 types on coord_ring, counter_circuit and forced_coord;
    - each with focal agents static, random, coop and fetcher, and 2 seeds.
- **Tests:** the 32 tests include golden tests that replay stored 300-step episodes from the
  original lock, redesign and graded runs action for action.

## What is known about these partners

On coord_ring and counter_circuit:

- **Speed of identification:** staging is identified in about 7–12 steps and ρ in 14–19. Soft
  role takes about 64–74 steps, and only when the focal agent fetches; the exclusive roles are
  identified by step 2.
- **Soft family:** knowing the type adds at most about 5% soups over the best type-agnostic
  strategy, on every layout tried. Fetching is the best committed role against all 12 types.
- **Exclusive family:** the type matters only if the focal agent must commit to a role before
  seeing the partner. Under a 300-step lock, knowing the type raises soups from 2.73 to 5.18,
  which is still below the 5.96 an unlocked agent gets without knowing it.
- **Graded family (coord_ring, 300-step lock; preliminary, not yet written up):** knowing the type
  is worth 0.3%, 3%, 5%, 48%, 88% and 90% of the type-agnostic agent's soups at q = 0, .25, .5,
  .75, .9 and 1. The best role already flips at q = 0.5, but the value appears only between 0.5
  and 0.75.
