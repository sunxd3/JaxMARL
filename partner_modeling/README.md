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

## Two more games, and agents for testing the settings

Added 2026-11-10. The Overcooked partner above is one case of a more general setting: a game,
another player whose policy is known exactly, and something hidden that the focal agent has to
infer from how that player behaves. This section adds two more games of that kind (a small
Hanabi and the Coin Game), writes the other players as probabilistic programs, and adds agents
whose job is to test whether a setting does what it is meant to. The agents are probes; their
scores are evidence about the games, not results in themselves.

### Files

The games and the other players:

| file | contents |
|---|---|
| `ppl.py` | a minimal probabilistic-programming layer: `sample`, `flip`, `observe`, `condition`, `factor`; `infer` (exact enumeration, or likelihood weighting) and `simulate`; `infer` calls nest |
| `pp_models.py` | Overcooked: the typed-intent partner as a program (`partner_action`; enumerating it reproduces `TypedIntentModel.dist`) and the belief step over (recipe, type, memory) |
| `observer.py: recipe_filter` | Overcooked: exact posterior over the current recipe from the partner's actions |
| `hanabi_model.py` | a small two-player Hanabi (3 colours, 3 cards each) as a plain-Python model, matched move for move to JaxMARL's `HanabiEnv`; `MiniHanabi` plays the real environment |
| `hanabi_partner.py` | the scripted Hanabi partner as a program, typed by its convention: what a hint means (newest / oldest / literal) and which card it discards |
| `coin_model.py` | the Coin Game as a plain-Python model matched to JaxMARL's `CoinGame`, and the partner (cooperator / greedy / tit-for-tat / grim) as a program |

Each other player is one program. Enumerating it gives the exact probability of every move;
simulating it plays the game.

Agents and runners, for testing the settings:

| file | contents |
|---|---|
| `pp_focal.py`, `pp_run.py` | Overcooked: an agent with an exact belief that plans by rollouts over short policies ("handle plates", "cook for recipe r", ...); runner with scripted baselines and the exact observers |
| `llm_focal.py`, `llm_run.py` | Overcooked: agents that ask `claude -p` at every decision; `ClaudeFocal` picks among valid tasks, `ClaudeStationFocal` picks any station and can have the recipe hidden |
| `hanabi_agents.py`, `hanabi_llm.py`, `hanabi_run.py` | Hanabi: the scripted partner as a player (`ConventionAgent`), an exact-belief player (`PPHanabiAgent`), an LLM player, and the runner |
| `coin_agents.py`, `coin_run.py` | Coin Game: scripted red policies, an exact-belief agent with a "probe" option, an LLM agent, and the runner |

The exact-belief agents enumerate the partner program for their belief and choose by imagined
rollouts. The Overcooked planner is rough: a hand-made set of options, a few rollouts per
recipe hypothesis, and it reuses the partner's own rules for which tasks are currently valid.

The LLM agents make one fresh `claude -p` call per decision. The prompt holds the rules, the
current state, the last 12 to 14 events, and a note of at most 40 words the model wrote at its
previous decision; there is no running conversation. In Overcooked the model picks a station or
task and a script walks there, in the Coin Game it picks which coin to go for, and in Hanabi it
picks the move itself. `--give-beliefs` writes the exact posterior into the prompt.

### The settings

- **Overcooked, recipe known only to the partner.** `asymm_advantages_recipes_center` has two
  pots and four possible recipes; a new recipe is drawn after every correct delivery and a wrong
  soup costs 20. With `--hide-recipe` the focal agent is not told the recipe and has to read it
  from which ingredients the partner fetches. The recipe is hidden by a flag on the agent, not by
  the game.
- **Hanabi, partner convention unknown.** Two players, 3 colours x 5 ranks, 3 cards each, 8
  hints, 3 lives, best score 15. The partner is one of six conventions: a hint means "play the
  newest card it touched", "play the oldest", or nothing beyond what it says, and it discards
  its oldest or its newest unhinted card. It makes a random move 5% of the time.
- **Coin Game, partner strategy unknown and reactive.** A 3x3 wrap-around grid; the focal agent
  is red. A cooperator, a tit-for-tat player and a grim player look the same until the focal
  agent steals a coin, and then react differently, so finding out has a price. The partner
  makes a random move 10% of the time.

### What the test runs say about each setting

Run on 2026-11-10. Samples are small: conditions with LLM calls have 1 to 12 episodes, the
others 3 to 48. Read the direction, not the decimals. The run records are under
`outputs/partner_modeling/`, which is not in the repository.

**Hanabi works as intended.**

| player | games | mean score of 15 | games that lost all lives |
|---|---|---|---|
| scripted pair, same hint meaning | 12 pairings x 20 | 7.8 | |
| scripted pair, different hint meanings | 24 pairings x 20 | 4.2 | |
| exact-belief player | 48 | 11.3 | 2 |
| exact-belief player, true hint meaning left out of its hypotheses | 48 | 8.0 | 8 |
| LLM (Opus), low effort | 12 | 6.1 | 5 |
| LLM (Opus), high effort | 6 | 11.0 | 0 |
| LLM (Opus), low effort, given the exact posterior | 12 | 9.3 | 1 |

- The hidden convention changes the right action: scripted pairs that disagree on what a hint
  means score about half of pairs that agree.
- Getting it wrong shows in the score: leaving the true hint meaning out of the exact-belief
  player's hypotheses costs about 3 points.
- It is not trivial to read. The LLM put 0.50 to 0.59 on the true hint meaning where the exact
  posterior had about 0.75, and at low effort 17 of its 65 plays were of a card the exact
  posterior put below one half.
- Losing all lives scores 0, so the mean is largely a count of lost games. The high-effort LLM
  already matches the exact-belief player, so there is little headroom at this size.

**The Coin Game works at 200 steps, not at the default 60.** Red's mean score against each
partner strategy:

| red player | 60 steps: cooperator / greedy / tit-for-tat | 200 steps: cooperator / greedy / tit-for-tat |
|---|---|---|
| scripted: red coins only | 36.8 / 9.7 / 36.8 | 125.0 / 26.1 / 125.0 |
| scripted: nearest coin | 43.8 / 20.0 / 16.1 | 148.0 / 57.2 / 61.0 |
| best fixed policy for the true strategy | 43.8 / 20.0 / 36.8 | 148.0 / 57.2 / 125.0 |
| exact-belief agent | 42.1 / 16.5 / 33.6 | 147.0 / 55.7 / 117.3 |
| LLM (Opus), low effort | 42.0 / 19.5 / 42.0 | 129 / 60 / 95 |
| exact-belief agent, tit-for-tat left out of its hypotheses | - / - / -4.3 | - / - / -19.2 |

- At 60 steps the hidden strategy never has to be worked out. The LLM never took a blue coin
  from a cooperator or a tit-for-tat player and did as well as the best fixed policy.
- At 200 steps working it out pays. The exact-belief agent steals once to find out and gets
  close to the best fixed policy against all three. The LLM, in its one episode against
  tit-for-tat, took 21 blue coins, lost 23 red ones in return, and scored 95 where never
  stealing gets 125.
- A wrong set of hypotheses is punished hard: told the partner is a cooperator or greedy, the
  exact-belief agent facing tit-for-tat concludes "cooperator" and scores below zero.

**In Overcooked the score is only partly about the hidden recipe.** Two-pot kitchen, partner
`runner_pot_direct_lo`, 200 steps:

| agent | recipe | episodes | mean reward | wrong soups per episode |
|---|---|---|---|---|
| LLM (Opus), low effort | hidden | 6 | +17 | 3.0 |
| LLM (Opus), high effort | hidden | 3 | +53 | 1.3 |
| LLM (Opus), low effort, given the exact posterior | hidden | 6 | +60 | 1.5 |
| LLM (Opus), low effort | shown | 3 | +60 | 2.0 |
| exact-belief agent | hidden | 8 | +60 | 0.9 |
| exact-belief agent | shown | 8 | +72 | 0.8 |
| scripted coop / scripted plater, both told the recipe | - | 8 | +10 / +25 | 3.8 / 2.5 |

- Hiding the recipe does cost the LLM. On the three seeds run both ways it scored +20, +20, +20
  with the recipe hidden and +100, +40, +40 with it shown.
- Most wrong soups are delivered by the partner, who knows the recipe. In the 18 LLM episodes
  there were 37 wrong deliveries: 31 by the partner, 5 by the LLM, 1 not traced. With the recipe
  shown, the LLM delivered no wrong soup and the partner delivered 6 in 3 episodes. In 8 fully
  logged episodes of the exact-belief agent, 6 of 7 wrong soups were the partner's.
- The cause is in the partner. It uses the recipe only to choose which ingredient pile to fetch
  from, and plates and delivers any cooked soup without checking it. It fills both pots for the
  current recipe; delivering the first redraws the recipe, and the second then often no longer
  matches.
- On a one-pot kitchen such as `cramped_room_v2` the hidden recipe never matters: the partner
  fills the only pot itself, and the LLM and two scripted players all scored +80. On those
  layouts the soft roles runner / plater / flex are also behaviourally identical, because
  fetching and plating never compete.
- On `coord_ring` with partners that keep strictly to one job, taking the same job as the
  partner scores 0, but the scripted coop policy does as well against both partner types (+105
  and +140) as the exact-belief agent (+100 and +145), so nothing has to be inferred there.

### Known weaknesses and open points

- **Overcooked partner and stale soups.** The partner should probably check a soup against the
  recipe before plating or delivering it, or the recipe should stay fixed for the episode.
- **The recipe is hidden in the agents**, not in what the game lets the focal agent observe.
- **Coin Game length.** 60 steps is the default and is too short; use `--steps 200`.
- **Hanabi scoring.** Counting cards played would keep one lost game from being a zero.
- **No standalone exact observers for Hanabi and the Coin Game**: the exact posteriors are
  inside `PPHanabiAgent` and `PPCoinAgent`. There are three runners and three agent interfaces.
- **One unexplained result.** The exact-belief Overcooked agent scored higher with a partner
  program that an LLM wrote from a verbal description (+88) than with the true one (+60) on the
  same 8 seeds. A rerun of one of those true-program episodes gave +40 where the saved run had
  +60, so the runs are also not exactly reproducible. Neither has been traced to a cause.
- **Failed LLM calls.** In 6 of the 12 low-effort Hanabi games and 6 of the 12 given-posterior
  games, 3 to 7 of about 20 turns fell back to a default move because the call failed. Three
  Coin Game episodes had 2 or 3 such turns out of 30.

### Running

```bash
python -m pytest tests/partner_modeling -q

# Hanabi: conventions against each other, the exact-belief player, the LLM player
python -m partner_modeling.hanabi_run cross --games 20 --out outputs/partner_modeling/hanabi/cross
python -m partner_modeling.hanabi_run pp --games 8 --out outputs/partner_modeling/hanabi/pp/all
python -m partner_modeling.hanabi_run llm --games 2 --partners 0 2 4 --model opus --out outputs/partner_modeling/hanabi/llm/all

# Coin Game: scripted policies against each partner, the exact-belief agent, the LLM
python -m partner_modeling.coin_run scripted --games 30 --out outputs/partner_modeling/coin/scripted
python -m partner_modeling.coin_run pp --games 10 --out outputs/partner_modeling/coin/pp
python -m partner_modeling.coin_run pp --games 3 --steps 200 --out outputs/partner_modeling/coin/long/pp
python -m partner_modeling.coin_run llm --games 2 --model opus --out outputs/partner_modeling/coin/llm/low

# Overcooked, exact-belief agent, recipe hidden (no paid calls)
python -m partner_modeling.pp_run --layout asymm_advantages_recipes_center --theta runner_pot_direct_lo \
    --seeds 0 1 2 --max-steps 200 --out outputs/partner_modeling/pp/asymm_hidden
# Overcooked, LLM (every decision is a paid call)
python -m partner_modeling.llm_run --layout asymm_advantages_recipes_center --hide-recipe --model opus \
    --max-steps 200 --out outputs/partner_modeling/llm/asymm_hidden_s0

# exact inference with the partner's true type left out of the hypotheses
python -m partner_modeling.hanabi_run pp --games 8 --exclude-true-signal --out outputs/partner_modeling/hanabi/pp_wrongset/all
python -m partner_modeling.coin_run pp --types cooperator greedy --partners tit_for_tat --out outputs/partner_modeling/coin/wrongset/s60
```
