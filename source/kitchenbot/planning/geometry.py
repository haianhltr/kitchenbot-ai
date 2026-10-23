"""Re-export of :mod:`kitchenbot.geometry`, which moved there from this module in #48.

The poses and quaternion helpers are shared by planning and controllers now, so they live at the
package root. This module keeps ``from kitchenbot.planning.geometry import ...`` working for #43's
planner tests and helpers; new code imports :mod:`kitchenbot.geometry` directly.
"""

from kitchenbot.geometry import *  # noqa: F403
from kitchenbot.geometry import __all__  # noqa: F401
