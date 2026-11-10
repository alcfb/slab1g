"""1-group, 1D slab k-eigenvalue: diffusion vs SP3 vs transport (even-parity S_N), with nutils."""
from .problem import SlabProblem, Zone
from .solvers import Result, diffusion, sp3, sn_even_parity, solve_all

__all__ = ["SlabProblem", "Zone", "Result", "diffusion", "sp3", "sn_even_parity", "solve_all"]
__version__ = "0.1.0"
