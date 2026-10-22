"""KitchenBot AI: humanoid kitchen manipulation in simulation.

Package layout follows docs/architecture.md section 15. Subpackages are populated by the issues
named in their docstrings; nothing here imports Isaac Sim at import time so that unit tests and CI
run without a GPU.
"""

__version__ = "0.1.0"

#: The only physics preset this project runs on. Every config and command must use it.
PHYSICS_PRESET = "isaacsim_physx"
