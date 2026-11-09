"""The Overcooked-V2 names the partner code uses, from ``jaxmarl.environments.overcooked_v2``.

``Actions`` is short for ``OvercookedActionsEnum``; ``state_time`` reads the state's timestep.
"""

from jaxmarl.environments.overcooked_v2.common import (  # noqa: F401
    Direction,
    DynamicObject,
    StaticObject,
)
from jaxmarl.environments.overcooked_v2.common import OvercookedActionsEnum as Actions  # noqa: F401


def state_time(s) -> int:
    """Current timestep of an (unwrapped) OvercookedV2 state."""
    return int(s.step)
