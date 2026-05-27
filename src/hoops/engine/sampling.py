"""RNG plumbing.

Single seeded ``numpy.random.Generator`` is threaded through the simulation
as an explicit argument to every sampling site. No module-level RNG, ever.
That's what makes seeded reproducibility actually hold under refactoring.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)
