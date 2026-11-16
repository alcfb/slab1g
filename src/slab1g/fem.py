"""Finite-element building blocks assembled with nutils.

Every angular approximation in this package (diffusion, SP3, even-parity S_N)
reduces to second-order equations of the form

    -d/dx ( a(x) du/dx ) + b(x) u = ...

with piecewise-constant coefficients. The weak forms therefore only need two
scalar operators per zone,

    K_z[i, j] = int_z  phi_i' phi_j' dx      (stiffness)
    M_z[i, j] = int_z  phi_i  phi_j  dx      (mass)

on one continuous Lagrange basis spanning the whole slab. Any coefficient-
weighted operator is then a linear combination sum_z c_z K_z (or M_z).
Continuity of the basis plus the natural (Neumann) weak form gives exactly the
interface conditions (continuous flux moments and currents) and the reflective
boundary conditions (zero odd moments / zero current), so no boundary terms
appear anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from nutils import function, mesh

from .problem import SlabProblem


def _csr(integral, n: int) -> sp.csr_matrix:
    values, rowptr, colidx = function.eval(function.as_csr(integral))
    return sp.csr_matrix((values, colidx, rowptr), shape=(n, n))


@dataclass
class Discretization:
    problem: SlabProblem
    nodes: np.ndarray
    zone_slices: list[slice]  # element ranges of each zone
    topo: object
    geom: object
    basis: object
    K: list[sp.csr_matrix]     # per-zone stiffness
    M: list[sp.csr_matrix]     # per-zone mass
    L: list[np.ndarray]        # per-zone load vector int_z phi_i dx

    @property
    def ndofs(self) -> int:
        return len(self.basis)

    # ---- coefficient-weighted operators ---------------------------------
    def stiffness(self, coef) -> sp.csr_matrix:
        """sum_z coef(z) K_z, `coef` a callable on Zone."""
        return sum(coef(z) * K for z, K in zip(self.problem.zones, self.K))

    def mass(self, coef) -> sp.csr_matrix:
        return sum(coef(z) * M for z, M in zip(self.problem.zones, self.M))

    # ---- post-processing --------------------------------------------------
    def zone_integrals(self, u: np.ndarray) -> np.ndarray:
        return np.array([Lz @ u for Lz in self.L])

    def total_integral(self, u: np.ndarray) -> float:
        return float(sum(self.zone_integrals(u)))

    def sample(self, u: np.ndarray, points_per_element: int = 9):
        """Evaluate the FE field u on a fine grid -> (x, u(x))."""
        smp = self.topo.sample("uniform", points_per_element)
        x = smp.eval(self.geom[0])
        f = smp.eval(function.dotarg("u", self.basis), arguments={"u": u})
        order = np.argsort(x, kind="stable")
        return x[order], f[order]


def build(problem: SlabProblem) -> Discretization:
    nodes = [0.0]
    zone_slices, start = [], 0
    for z in problem.zones:
        ne = max(problem.min_elements_per_zone,
                 math.ceil(z.thickness * z.sigma_t * problem.elements_per_mfp))
        nodes.extend(np.linspace(nodes[-1], nodes[-1] + z.thickness, ne + 1)[1:])
        zone_slices.append(slice(start, start + ne))
        start += ne
    nodes = np.asarray(nodes)

    topo, geom = mesh.rectilinear([nodes])
    basis = topo.basis("std", degree=problem.degree)
    n = len(basis)
    dbasis = function.grad(basis, geom)[:, 0]
    # Cylindrical geometry (radial coordinate r = x, symmetry axis at x = 0): every
    # volume integral carries the weight r (per radian and unit height), so
    # K_z = int r phi_i' phi_j' dr etc. The natural boundary condition at r = 0 is the
    # symmetry condition, at r = R zero net current, exactly as in the slab.
    cyl = getattr(problem, "geometry", "slab") == "cylinder"
    deg = 2 * problem.degree + (1 if cyl else 0)
    J = function.J(geom) * (geom[0] if cyl else 1)

    K, M, L = [], [], []
    for s in zone_slices:
        sub = topo[s]
        K.append(_csr(sub.integral(dbasis[:, None] * dbasis[None, :] * J, degree=deg), n))
        M.append(_csr(sub.integral(basis[:, None] * basis[None, :] * J, degree=deg), n))
        L.append(function.eval(sub.integral(basis * J, degree=deg)))

    return Discretization(problem, nodes, zone_slices, topo, geom, basis, K, M, L)
