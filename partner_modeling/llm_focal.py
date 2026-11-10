"""LLM focal agents driven by ``claude -p`` (Claude Code in headless mode).

Two agents share the prompt and logging code:

* ``ClaudeFocal`` picks a *task*: one of the focal agent's currently valid intents (the ones the
  scripted focals use) or ``wait``.  The feature extractor filters the options and the controller
  picks the target, so little is left to get wrong.
* ``ClaudeStationFocal`` picks a *station* to walk to and interact with: any ingredient pile,
  pot, counter, the plate pile or the delivery window, useful or not.  It chooses which
  ingredient and which pot.  With ``hide_recipe`` it is not told the current recipe (the scripted
  partner always knows it) and has to read it from what the partner does.

Neither picks primitive moves; the shared deterministic controller walks.  The model is asked
again only at decision points, so an episode costs a few dozen calls instead of one per step.
Each call is stateless: the prompt holds a text view of the kitchen, a log of recent events, and
the short ``notes`` the model wrote at its previous decision (its only memory).  Besides its
choice the model reports beliefs (partner role and staging, or the recipe) that ``llm_run.py``
compares with the exact observers.

The model is told the partner's type space and behaviour rule in words, not the likelihood.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from partner_modeling.env_compat import Actions, DynamicObject, StaticObject
from partner_modeling.partners.typed_intent import (
    INTENT_ADD_INGREDIENT, INTENT_DELIVER, INTENT_FETCH_PILE, INTENT_FETCH_STAGED,
    INTENT_GET_PLATE, INTENT_IDLE, INTENT_NAMES, INTENT_PICKUP_SOUP, INTENT_STAGE, INTENT_YIELD,
    STAGING_NAMES, PartnerView, TypedIntentConfig, TypedIntentModel, uniform_from_key,
    unwrap_env_state,
)
from partner_modeling.pp_models import ingredients_of
from partner_modeling.rollouts import FocalPolicy, _fwd

WAIT = "wait"
TASKS = (INTENT_DELIVER, INTENT_PICKUP_SOUP, INTENT_GET_PLATE, INTENT_ADD_INGREDIENT,
         INTENT_FETCH_PILE, INTENT_FETCH_STAGED, INTENT_STAGE)
TASK_TEXT = {
    INTENT_DELIVER: "take the soup you hold to the delivery window (or collect a dish left on a counter)",
    INTENT_PICKUP_SOUP: "with the plate you hold, take the soup out of a pot (waits beside the pot while it cooks)",
    INTENT_GET_PLATE: "fetch a plate",
    INTENT_ADD_INGREDIENT: "put the ingredient you hold into a pot",
    INTENT_FETCH_PILE: "take an ingredient the recipe needs from its pile",
    INTENT_FETCH_STAGED: "pick up an ingredient that is lying on a counter",
    INTENT_STAGE: "put what you hold on a counter",
}
# Task agent's fallback when a call fails or returns an invalid choice: the scripted "coop" weights.
COOP_WEIGHTS = {INTENT_DELIVER: 10, INTENT_PICKUP_SOUP: 8, INTENT_GET_PLATE: 4, INTENT_ADD_INGREDIENT: 6,
                INTENT_FETCH_PILE: 4, INTENT_FETCH_STAGED: 4.5, INTENT_STAGE: 1}

# Role labels the model reports, and how each family's roles are described to it.
ROLE_LABEL = {0: "runner", 1: "plater", 2: "flex", 3: "runner", 4: "plater", 5: "runner", 6: "plater"}
ROLE_TEXT = {
    "typed_intent": {
        "runner": "strongly prefers fetching ingredients to fetching plates",
        "plater": "strongly prefers fetching plates and serving to fetching ingredients",
        "flex": "has no preference between the two",
    },
    "typed_intent_x": {
        "runner": "only handles ingredients; it never fetches a plate, so it never serves a soup",
        "plater": "only fetches plates, serves and delivers; it never fetches an ingredient",
    },
    "typed_intent_g": {
        "runner": "mostly handles ingredients and fetches a plate only some of the time",
        "plater": "mostly fetches plates and serves, and fetches an ingredient only some of the time",
    },
}
DIR_NAMES = ("up", "down", "right", "left")
INGREDIENT_NAMES = ("onion", "tomato", "pepper", "mushroom")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def recipe_code(recipe: Sequence[int]) -> str:
    """[0, 0, 1] -> 'OOT' (first letters of the ingredient names, sorted by ingredient)."""
    return "".join(INGREDIENT_NAMES[i][0].upper() for i in sorted(int(i) for i in recipe))


def _names(idx: Sequence[int]) -> str:
    return ", ".join(INGREDIENT_NAMES[i] for i in idx) or "empty"


def _item(v: int) -> str:
    v = int(v)
    if v == 0:
        return "nothing"
    if v == DynamicObject.PLATE:
        return "a plate"
    if v & DynamicObject.PLATE:
        return f"a plated soup ({_names(ingredients_of(v))})"
    ing = ingredients_of(v)
    if len(ing) == 1:
        return ("an " if INGREDIENT_NAMES[ing[0]][0] in "aeiou" else "a ") + INGREDIENT_NAMES[ing[0]]
    return f"a soup ({_names(ing)})"


def _normalise(d: Any, labels) -> Optional[Dict[str, float]]:
    """Clip to >= 0 and renormalise a reported distribution; None if it is unusable."""
    try:
        v = np.array([max(float(d[k]), 0.0) for k in labels])
    except (KeyError, TypeError, ValueError):
        return None
    return {k: float(x) for k, x in zip(labels, v / v.sum())} if v.sum() > 0 else None


# ---------------------------------------------------------------------------
# claude -p
# ---------------------------------------------------------------------------

class ClaudeCLI:
    """One stateless ``claude -p`` call per question, with a JSON schema for the reply."""

    def __init__(self, model: Optional[str] = None, effort: Optional[str] = "low", timeout: float = 180.0,
                 retries: int = 1):
        self.model, self.effort, self.timeout, self.retries = model, effort, timeout, retries
        # An empty working directory, so no project instructions or memory reach the model.
        self.cwd = tempfile.mkdtemp(prefix="claude_focal_")
        # Every prompt is new, so a cache write is never read back; writing one costs more than
        # plain input (measured: about 40% of the cost of a call).
        self.env = {**os.environ, "DISABLE_PROMPT_CACHING": "1"}

    def ask(self, system: str, prompt: str, schema: Dict[str, Any]) -> Tuple[Optional[dict], Dict[str, Any]]:
        """Returns (reply or None, meta).  meta: latency_s, cost_usd, model, error."""
        cmd = ["claude", "-p", "--output-format", "json", "--no-session-persistence", "--safe-mode",
               "--strict-mcp-config", "--disable-slash-commands", "--tools", "",
               "--system-prompt", system, "--json-schema", json.dumps(schema)]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        meta: Dict[str, Any] = {"latency_s": 0.0, "cost_usd": 0.0, "model": None, "error": None}
        for _ in range(self.retries + 1):
            t0 = time.time()
            try:
                p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout,
                                   cwd=self.cwd, env=self.env)
                out = json.loads(p.stdout)
            except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
                meta["latency_s"] += time.time() - t0
                meta["error"] = f"{type(e).__name__}: {e}"[:300]
                continue
            meta["latency_s"] += time.time() - t0
            meta["cost_usd"] += float(out.get("total_cost_usd") or 0.0)
            meta["model"] = next(iter(out.get("modelUsage") or {}), None)
            if out.get("is_error"):
                meta["error"] = str(out.get("result") or out.get("subtype"))[:300]
                continue
            reply = out.get("structured_output")
            if reply is None:
                try:
                    reply = json.loads(out.get("result") or "")
                except json.JSONDecodeError:
                    meta["error"] = "reply is not JSON"
                    continue
            meta["error"] = None
            return reply, meta
        return None, meta


# ---------------------------------------------------------------------------
# Shared agent code
# ---------------------------------------------------------------------------

class _ClaudeFocalBase(FocalPolicy):
    """Event tracking, kitchen text, the ``claude -p`` round trip and the decision log.

    ``records`` holds one dict per decision: step, offered options, choice, reported beliefs,
    notes, latency, cost and a fallback flag (the call failed or the choice was not offered).
    """

    system = ""
    reasoning = False        # ask for a short "reasoning" field before the choice
    show_recipe = False      # print the current recipe in the kitchen text

    def __init__(self, layout, model: Optional[str] = None, effort: Optional[str] = "low", max_hold: int = 8,
                 max_steps: Optional[int] = None, ctrl_cfg: Optional[Dict] = None, yield_prob: float = 0.15,
                 timeout: float = 180.0, log_path: Optional[str] = None, verbose: bool = False):
        self.model = TypedIntentModel(layout, agent_id=0,
                                      config=TypedIntentConfig(controller="aware", **(ctrl_cfg or {})))
        self.cli = ClaudeCLI(model=model, effort=effort, timeout=timeout)
        self.max_hold, self.max_steps, self.yield_prob = int(max_hold), max_steps, yield_prob
        self.log_path, self.verbose = log_path, verbose
        self.beliefs: Dict[str, Tuple[str, ...]] = {}     # reply field -> labels of the distribution
        self.reset(None)

    def reset(self, key):
        self._t = 0
        self._prev: Optional[PartnerView] = None
        self._cur: Optional[Any] = None
        self._t_query = -10 ** 6
        self._notes = ""
        self._events: List[Dict[str, Any]] = []      # {t, agent, text, kind}
        self._still: List[bool] = []                 # partner unchanged between consecutive views
        self._choices: List[Tuple[int, str]] = []
        self._last_a: Optional[int] = None
        self._blocked = 0                            # consecutive move actions that did not move me
        self.records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ tracking
    def _track(self, view: PartnerView, env_state):
        """Inventory events of both agents, and whether the partner changed at all, since the last view."""
        p = self._prev
        if p is None:
            return
        moved = not np.array_equal(p.pos[0], view.pos[0])
        self._blocked = self._blocked + 1 if (self._last_a is not None and self._last_a < 4 and not moved) else 0
        for ag in (0, 1):
            v0, v1 = int(p.inv[ag]), int(view.inv[ag])
            if v0 == v1:
                continue
            fy, fx = _fwd(view, ag)
            inside = 0 <= fy < p.grid.shape[0] and 0 <= fx < p.grid.shape[1]
            st = int(p.grid[fy, fx, 0]) if inside else -1
            at, kind = f"({fy},{fx})", "other"
            if st == StaticObject.POT and v1 == 0:
                text = f"put {_item(v0)} in the pot at {at}"
            elif st == StaticObject.POT:
                text = f"took the soup ({_names(ingredients_of(v1))}) out of the pot at {at}"
            elif st == StaticObject.GOAL and v1 == 0:
                ok = bool(np.asarray(getattr(unwrap_env_state(env_state), "new_correct_delivery", True)))
                kind = "delivery_ok" if ok else "delivery_wrong"
                text = f"delivered a soup ({_names(ingredients_of(v0))}): " + ("correct, +20" if ok else "WRONG, -20")
            elif st == StaticObject.PLATE_PILE:
                text = "took a plate from the plate pile"
            elif st >= StaticObject.INGREDIENT_PILE_BASE:
                text = f"took {_item(v1)} from its pile"
            elif st == StaticObject.WALL and v1 == 0:
                text = f"put {_item(v0)} on the counter at {at}"
            elif st == StaticObject.WALL:
                text = f"picked up {_item(v1)} from the counter at {at}"
            else:
                text = f"went from holding {_item(v0)} to holding {_item(v1)}"
            self._events.append({"t": self._t - 1, "agent": ag, "text": text, "kind": kind})
        self._still.append(bool(np.array_equal(p.pos[1], view.pos[1]) and p.dir[1] == view.dir[1]
                                and p.inv[1] == view.inv[1]))

    def _n(self, kind: str) -> int:
        return sum(e["kind"] == kind for e in self._events)

    def _emit(self, a: int, f, view: PartnerView, key) -> int:
        """Low-level collision handling around the chosen action.  As in the scripted focals, stay
        with prob. ``yield_prob`` when a move may be contested.  Two cooks stepping into the same
        cell block each other every step, so after two blocked moves also stay with prob. 0.5."""
        u = uniform_from_key(key)
        if f.valid[INTENT_YIELD] and u < self.yield_prob:
            a = f.ctrl[INTENT_YIELD]
        elif self._blocked >= 2 and u < 0.5:
            a = Actions.stay
        self._last_a, self._prev = int(a), view
        self._t += 1
        return int(a)

    # ------------------------------------------------------------ prompt pieces
    def _kitchen(self, view: PartnerView) -> List[str]:
        t, T = self._t, self.max_steps
        ok, bad = self._n("delivery_ok"), self._n("delivery_wrong")
        pos = [tuple(int(c) for c in view.pos[a]) for a in (0, 1)]
        pots, items = [], []
        for (y, x) in map(tuple, np.argwhere(view.grid[:, :, 0] == StaticObject.POT)):
            dyn, timer = int(view.grid[y, x, 1]), int(view.grid[y, x, 2])
            ing = ingredients_of(dyn)
            state = (f"soup ready ({_names(ing)})" if dyn & DynamicObject.COOKED
                     else f"cooking ({_names(ing)}), {timer} steps left" if timer > 0
                     else f"holds {_names(ing)} ({len(ing)}/3)" if ing else "empty")
            pots.append(f"({y},{x}): {state}")
        for (y, x) in map(tuple, np.argwhere((view.grid[:, :, 0] == StaticObject.WALL) & (view.grid[:, :, 1] != 0))):
            items.append(f"{_item(int(view.grid[y, x, 1]))} at ({y},{x})")
        out = [f"Step {t}" + (f" of {T}" if T else "") + f". Score so far: {20 * (ok - bad):+d} "
               f"({ok} correct and {bad} wrong deliveries)."]
        if self.show_recipe:
            r = ingredients_of(view.recipe)
            out.append(f"Current recipe: {recipe_code(r)} ({_names(r)}).")
        out += ["", "Kitchen (r = row from the top, c = column from the left):", self._grid(view), "",
                f"YOU: at {pos[0]}, facing {DIR_NAMES[int(view.dir[0])]}, holding {_item(view.inv[0])}."
                + (f" Your last {self._blocked} moves were blocked." if self._blocked >= 2 else ""),
                f"PAR: at {pos[1]}, facing {DIR_NAMES[int(view.dir[1])]}, holding {_item(view.inv[1])}.",
                "Pots: " + "; ".join(pots) + ".",
                "Items on counters: " + ("; ".join(items) if items else "none") + "."]
        return out

    def _history(self, n_events: int = 14) -> List[str]:
        out = ["Recent events (most recent last):"]
        out += [f"  step {e['t']}: {'YOU' if e['agent'] == 0 else 'PAR'} {e['text']}"
                for e in self._events[-n_events:]] or ["  nothing yet"]
        if self._still:
            recent = self._still[-20:]
            out.append(f"PAR did not move, turn or handle anything on {sum(recent)} of the last {len(recent)} "
                       f"steps ({sum(self._still)} of {len(self._still)} overall).")
        if self._choices:
            out.append("Your last choices: " + ", ".join(f"step {s} {c}" for s, c in self._choices[-5:]) + ".")
        out += ["", f'Your notes from your last decision: "{self._notes}"' if self._notes
                else "You have no notes yet."]
        return out

    @staticmethod
    def _grid(view: PartnerView) -> str:
        H, W = view.grid.shape[:2]
        rows = ["     " + "".join(f"c{x:<4d}" for x in range(W))]
        piles = set()
        for y in range(H):
            cells = []
            for x in range(W):
                st, dyn = int(view.grid[y, x, 0]), int(view.grid[y, x, 1])
                if tuple(view.pos[0]) == (y, x):
                    c = "YOU"
                elif tuple(view.pos[1]) == (y, x):
                    c = "PAR"
                elif st == StaticObject.POT:
                    c = "POT"
                elif st == StaticObject.GOAL:
                    c = "OUT"
                elif st == StaticObject.PLATE_PILE:
                    c = "PLT"
                elif st >= StaticObject.INGREDIENT_PILE_BASE:
                    name = INGREDIENT_NAMES[st - StaticObject.INGREDIENT_PILE_BASE]
                    piles.add(name)
                    c = name[:3].upper()
                elif st == StaticObject.WALL:
                    c = "#*" if dyn else "#"
                elif st == StaticObject.EMPTY:
                    c = "."
                else:
                    c = "x"
                cells.append(f"{c:<5s}")
            rows.append(f"r{y:<4d}" + "".join(cells).rstrip())
        legend = ("  # counter (#* = holding an item), POT pot, "
                  + ", ".join(f"{n[:3].upper()} {n} pile" for n in sorted(piles))
                  + ", PLT plate pile, OUT delivery window, . floor, x blocked")
        return "\n".join(rows + [legend])

    # ------------------------------------------------------------ the call
    def _query(self, prompt: str, names: List[str], fallback_choice: str) -> str:
        props: Dict[str, Any] = {}
        if self.reasoning:
            props["reasoning"] = {"type": "string"}
        for field, labels in self.beliefs.items():
            props[field] = {"type": "object", "additionalProperties": False, "required": list(labels),
                            "properties": {k: {"type": "number"} for k in labels}}
        props["choice"] = {"type": "string", "enum": names}
        props["notes"] = {"type": "string"}
        schema = {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}
        reply, meta = self.cli.ask(self.system, prompt, schema)
        choice = reply.get("choice") if isinstance(reply, dict) else None
        fallback = choice not in names
        if fallback:
            choice, reply = fallback_choice, {}
        else:
            self._notes = str(reply.get("notes", ""))[:400]
        self._t_query = self._t
        self._choices.append((self._t, choice))
        rec = {"t": self._t, "offered": names, "choice": choice, "fallback": fallback,
               "beliefs": {f: _normalise(reply.get(f), labels) for f, labels in self.beliefs.items()},
               "reasoning": reply.get("reasoning", ""), "notes": reply.get("notes", ""), **meta}
        self.records.append(rec)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps({**rec, "prompt": prompt}) + "\n")
        if self.verbose:
            b = next(iter(rec["beliefs"].values()), None)
            print(f"  t={self._t:3d} {choice:14s} {b} {meta['latency_s']:.1f}s"
                  + ("  [fallback: %s]" % meta["error"] if fallback else ""), flush=True)
        return choice

    def _type_text(self, family: str) -> str:
        roles = ROLE_TEXT[family]
        return ("It has a hidden type. Role: " + "; ".join(f'"{k}" {v}' for k, v in roles.items())
                + '. Staging: "pot_direct" carries ingredients straight to a pot; "counter_stager" prefers to '
                  "leave them on a counter between the piles and the pots for someone else to pick up.")


# ---------------------------------------------------------------------------
# Task agent
# ---------------------------------------------------------------------------

TASK_SYSTEM = """\
You control one of two cooks (YOU) in a small grid kitchen. The other cook (PAR) is a scripted \
partner. You share one score: the number of soups delivered before the episode ends.

How a soup is made: take ingredients from a pile and put 3 into a pot. The pot then cooks by \
itself for 20 steps. When the soup is ready, a cook holding a plate takes it out of the pot and \
carries it to the delivery window. A cook holds at most one item. Items can be put on counters and \
picked up again by either cook. Cooks cannot walk through each other and corridors are one cell \
wide, so being in the partner's way costs time.

You do not choose individual moves. You choose a task; a controller walks you there and carries \
it out, one step at a time. You are asked again whenever your set of available tasks changes, and \
at least every {max_hold} steps. Only tasks that are possible and currently useful are offered. \
"wait" (step out of the way and stand still) is always offered.

The partner's behaviour is fixed for the whole episode. It sees the kitchen and what you hold, \
but it does not adapt to you. {type_text}
It also pauses at random (some partners often) and now and then makes a random move.

Work out the partner's type from what it does, and take on the work it leaves undone.

Reply with:
- partner_role: your probability for each role ({labels}), summing to 1.
- partner_staging: your probability for pot_direct and counter_stager, summing to 1.
- choice: one of the offered task names.
- notes: at most 40 words. This is your only memory; it is shown to you at your next decision."""


class ClaudeFocal(_ClaudeFocalBase):
    """Focal agent (agent_0) that picks among its currently valid tasks.  ``family`` selects how
    the partner's roles are described; ``model`` / ``effort`` go to the CLI (``model=None`` uses
    the CLI's default model)."""

    name = "claude_tasks"

    def __init__(self, layout, family: str = "typed_intent", **kw):
        super().__init__(layout, **kw)
        labels = tuple(ROLE_TEXT[family])
        self.beliefs = {"partner_role": labels, "partner_staging": tuple(STAGING_NAMES)}
        self.system = TASK_SYSTEM.format(max_hold=self.max_hold, labels=" / ".join(labels),
                                         type_text=self._type_text(family))

    def reset(self, key):
        super().reset(key)
        self._offered: Tuple[int, ...] = ()

    def act(self, view: PartnerView, env_state, key) -> int:
        self._track(view, env_state)
        f = self.model.features(view)
        offered = tuple(i for i in TASKS if f.valid[i])
        if self._cur is None or offered != self._offered or self._t - self._t_query >= self.max_hold:
            self._decide(view, f, offered)
        return self._emit(f.ctrl[INTENT_IDLE if self._cur == WAIT else self._cur], f, view, key)

    def _decide(self, view: PartnerView, f, offered: Tuple[int, ...]):
        names = [INTENT_NAMES[i] for i in offered] + [WAIT]
        out = self._kitchen(view) + [""] + self._history() + ["", "Tasks you can choose now:"]
        for i in offered:
            extra = " (nothing else can be done with it right now)" if i == INTENT_STAGE and f.forced_stage else ""
            out.append(f"- {INTENT_NAMES[i]} (about {int(f.dist[i])} steps away): {TASK_TEXT[i]}{extra}")
        out.append(f"- {WAIT}: step out of the way and stand still")
        ok = [i for i in offered if i in COOP_WEIGHTS]
        fallback = (INTENT_NAMES[max(ok, key=lambda i: COOP_WEIGHTS[i] * f.value[i] - 0.5 * f.dist[i])]
                    if ok else WAIT)
        choice = self._query("\n".join(out), names, fallback)
        self._cur = WAIT if choice == WAIT else INTENT_NAMES.index(choice)
        self._offered = offered


# ---------------------------------------------------------------------------
# Station agent
# ---------------------------------------------------------------------------

STATION_SYSTEM = """\
You control one of two cooks (YOU) in a small grid kitchen. The other cook (PAR) is a scripted \
partner. You share one score: +20 for each delivered soup that matches the current recipe, -20 \
for each delivered soup that does not.

Recipes: a soup is 3 ingredients. {recipe_text}

How a soup is made: take ingredients from the piles and put 3 into a pot. Once a pot holds 3 \
ingredients it cooks by itself for 20 steps, whatever they are; nothing can be taken back out of \
a pot except the finished soup, with a plate. A cook holding a plate takes a ready soup out of the \
pot and carries it to the delivery window. A cook holds one item at a time. Items can be put on \
free counters and picked up again by either cook. Cooks cannot walk through each other and the \
kitchen is tight, so standing in the partner's way costs time.

You do not choose individual moves. You choose a station; a controller walks you there, turns you \
to face it and interacts once. What the interaction does depends on what you hold:
- ingredient pile or plate pile, holding nothing: take one.
- pot, holding an ingredient: add it, if the pot is not full or cooking. Holding a plate, and the \
soup is ready: take the soup.
- delivery window, holding a plated soup: deliver it.
- counter: put down what you hold if the counter is free; pick up its item if you hold nothing.
Anything else does nothing and wastes the step. Every station is always offered, useful or not. \
"wait" steps out of the way and stands still.
You are asked again after each interaction, whenever a pot or a counter changes, after a delivery, \
and at least every {max_hold} steps.

PAR's behaviour rule is fixed for the whole episode. It sees the kitchen and what you hold, but \
does not otherwise adapt to you. {partner_recipe_text}It adds what it carries to any pot with room, \
and it does not check a pot's contents before plating and delivering a soup. It pauses at random \
and now and then makes a random move. {type_text}

Reply with:
- reasoning: at most 50 words.
{belief_text}- choice: one of the offered station names.
- notes: at most 40 words. This is your only memory; it is shown to you at your next decision."""


class ClaudeStationFocal(_ClaudeFocalBase):
    """Focal agent (agent_0) that picks any station to interact with.

    ``hide_recipe``: do not tell the model the current recipe (multi-recipe layouts only).
    ``ask``: belief fields to request; default ``("recipe",)`` when the recipe is hidden,
    otherwise ``("partner_role", "partner_staging")``.
    """

    name = "claude_stations"
    reasoning = True

    def __init__(self, layout, family: str = "typed_intent", hide_recipe: bool = False,
                 ask: Optional[Sequence[str]] = None, belief_tracker=None, **kw):
        """``belief_tracker``: a ``pp_focal.PPFocal`` that follows the same episode; its exact
        recipe posterior is then put in the prompt, so that only the decision is the model's."""
        super().__init__(layout, **kw)
        self.tracker = belief_tracker
        self.recipes = [sorted(int(i) for i in r) for r in np.asarray(layout.possible_recipes).tolist()]
        self.recipe_codes = tuple(recipe_code(r) for r in self.recipes)
        multi = len(self.recipes) > 1
        self.hide_recipe = bool(hide_recipe and multi)
        self.show_recipe = multi and not self.hide_recipe
        ask = tuple(ask) if ask else (("recipe",) if self.hide_recipe else ("partner_role", "partner_staging"))
        labels = {"recipe": self.recipe_codes, "partner_role": tuple(ROLE_TEXT[family]),
                  "partner_staging": tuple(STAGING_NAMES)}
        self.beliefs = {k: labels[k] for k in ask}
        listing = "; ".join(f"{c} = {_names(r)}" for c, r in zip(self.recipe_codes, self.recipes))
        if not multi:
            recipe_text = f"The recipe is always {listing}."
            partner_recipe = ""
        else:
            recipe_text = (f"The possible recipes are: {listing}. One is drawn at random at the start, and a new "
                           "one is drawn at random (it may repeat) each time a correct soup is delivered. "
                           + ("You cannot see the current recipe. PAR always knows it." if self.hide_recipe
                              else "You and PAR both always know the current recipe."))
            partner_recipe = ("From the piles it takes only ingredients that the current recipe still needs, given "
                              "what is already in a pot with room; when more than one kind is still needed it "
                              "goes to the nearest such pile. It may also pick up any ingredient lying on a "
                              "counter, whatever it is. ")
        belief_text = "".join({
            "recipe": f"- recipe: your probability for each possible current recipe ({' / '.join(self.recipe_codes)}), "
                      "summing to 1.\n",
            "partner_role": f"- partner_role: your probability for each role ({' / '.join(labels['partner_role'])}), "
                            "summing to 1.\n",
            "partner_staging": "- partner_staging: your probability for pot_direct and counter_stager, summing to 1.\n",
        }[k] for k in ask)
        self.system = STATION_SYSTEM.format(max_hold=self.max_hold, recipe_text=recipe_text,
                                            partner_recipe_text=partner_recipe, belief_text=belief_text,
                                            type_text=self._type_text(family))

    def reset(self, key):
        super().reset(key)
        self._sig: Any = None
        self._interacted = False

    def _stations(self, view: PartnerView) -> Dict[str, np.ndarray]:
        """Station name -> target mask.  Nothing is filtered by usefulness."""
        m, st, dyn = self.model, view.grid[:, :, 0], view.grid[:, :, 1]
        out: Dict[str, np.ndarray] = {}

        def one(y, x):
            mask = np.zeros(st.shape, dtype=bool)
            mask[y, x] = True
            return mask

        for i, name in enumerate(INGREDIENT_NAMES):
            mask = st == StaticObject.INGREDIENT_PILE_BASE + i
            if mask.any():
                out[f"{name}_pile"] = mask
        out["plate_pile"] = m.plate_pile_mask_np
        for (y, x) in map(tuple, np.argwhere(m.pot_mask_np)):
            out[f"pot_{y}_{x}"] = one(y, x)
        out["delivery"] = m.goal_mask_np
        for (y, x) in map(tuple, np.argwhere(m.accessible_counter_mask_np & (dyn != 0))):
            out[f"counter_{y}_{x}"] = one(y, x)
        free = m.accessible_counter_mask_np & (dyn == 0)
        if free.any():
            out["free_counter"] = free
        return out

    def _signature(self, view: PartnerView):
        """What triggers a new decision when it changes: pot contents and state, counter items, deliveries."""
        st = view.grid[:, :, 0]
        pots = tuple((int(view.grid[y, x, 1]), bool(view.grid[y, x, 2] > 0))
                     for (y, x) in map(tuple, np.argwhere(st == StaticObject.POT)))
        ctr = tuple(int(v) for v in view.grid[:, :, 1][st == StaticObject.WALL])
        return pots, ctr, self._n("delivery_ok") + self._n("delivery_wrong")

    def act(self, view: PartnerView, env_state, key) -> int:
        self._track(view, env_state)
        if self.tracker is not None:            # it updates its belief from what the partner just did
            if self._last_a is not None:
                self.tracker._last_a = self._last_a
            self.tracker.act(view, env_state, key)
        stations, sig = self._stations(view), self._signature(view)
        py, px, pdir = int(view.pos[0, 0]), int(view.pos[0, 1]), int(view.dir[0])
        other = np.array([[int(view.pos[1, 0]), int(view.pos[1, 1])]], dtype=np.int32)
        ctrl = {k: self.model._ctrl_to_target(mask, py, px, pdir, other) for k, mask in stations.items()}
        ctrl = {k: v for k, v in ctrl.items() if v[0] is not None}
        if (self._cur is None or self._interacted or sig != self._sig
                or self._t - self._t_query >= self.max_hold or (self._cur != WAIT and self._cur not in ctrl)):
            self._decide(view, ctrl)
            self._sig = sig
        f = self.model.features(view)       # used only for the recipe-independent yield and park moves
        a = self._emit(f.ctrl[INTENT_IDLE] if self._cur == WAIT else ctrl[self._cur][0], f, view, key)
        self._interacted = a == int(Actions.interact)
        return a

    def _decide(self, view: PartnerView, ctrl: Dict[str, Tuple[int, float]]):
        dyn = view.grid[:, :, 1]
        out = self._kitchen(view)
        if self.hide_recipe:
            out.insert(1, "The current recipe is hidden from you.")
        if self.tracker is not None and self.tracker.trace:
            from partner_modeling.pp_models import encode_recipe
            post = self.tracker.trace[-1]["recipe"]
            out.insert(2, "An exact inference engine, given everything PAR has done, computes these recipe "
                          "probabilities: " + ", ".join(f"{c} {post.get(encode_recipe(r), 0.0):.2f}"
                                                        for c, r in zip(self.recipe_codes, self.recipes))
                          + ". These are correct; you may rely on them.")
        out += [""] + self._history() + ["", "Stations you can choose now:"]
        for k, (_, d) in ctrl.items():
            what = ""
            if k.startswith("counter_"):
                y, x = (int(v) for v in k.split("_")[1:])
                what = f", holds {_item(int(dyn[y, x]))}"
            elif k == "free_counter":
                what = ", the nearest free counter"
            out.append(f"- {k} (about {int(d)} steps away{what})")
        out.append(f"- {WAIT}")
        self._cur = self._query("\n".join(out), list(ctrl) + [WAIT], WAIT)
