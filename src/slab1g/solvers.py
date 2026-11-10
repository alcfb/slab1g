r"""One-group k-eigenvalue solvers for a reflected multi-zone slab.

All three models are written as a generalized eigenproblem

    A u = (1/k) F u

and solved for the fundamental mode (largest k) with ARPACK on A^{-1} F.

Notation per zone: St = total, Ss = scattering, Sa = St - Ss = absorption,
nSf = nu * fission.

Diffusion (P1)
--------------
    -(D phi')' + Sa phi = (1/k) nSf phi,          D = 1/(3 St)

SP3 (Gelbard; Brantley & Larsen form, isotropic scattering)
-----------------------------------------------------------
Unknowns Phi0 = phi0 + 2 phi2 and phi2; scalar flux phi = Phi0 - 2 phi2.

    -(1/(3St) Phi0')' + Sa (Phi0 - 2phi2)                   = (1/k) nSf (Phi0 - 2phi2)
    -(9/(35St) phi2')' + St phi2 - 2/5 Sa (Phi0 - 2phi2)    = -2/5 (1/k) nSf (Phi0 - 2phi2)

In 1D planar geometry SP3 is identical to P3, so this is a genuine angular
refinement of diffusion. Reflective BCs (phi1 = phi3 = 0) and P3 interface
conditions are both natural in this weak form.

Transport (even-parity S_N)
---------------------------
With psi+(x, mu) = [psi(x, mu) + psi(x, -mu)] / 2 for Gauss-Legendre directions
mu_m > 0 (full-range weights w_m, sum over all = 2):

    -(mu_m^2/St psi_m')' + St psi_m = 1/2 (Ss + nSf/k) phi,  phi = sum_m 2 w_m psi_m

Reflection psi(mu) = psi(-mu) means zero odd parity, i.e. a natural BC. As N
grows this converges to the exact transport solution; N = 16-32 is the
reference in the comparison. An independent diamond-difference S_N code in
`reference.py` is used in the tests to cross-check it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fem import Discretization, build
from .problem import SlabProblem


@dataclass
class Result:
    method: str
    k: float
    x: np.ndarray            # sample points
    phi: np.ndarray          # scalar flux at x, normalised to slab-average 1
    zone_avg: np.ndarray     # zone-averaged scalar flux (same normalisation)
    ndofs: int
    info: dict = field(default_factory=dict)

    def summary(self) -> str:
        za = ", ".join(f"{v:.5f}" for v in self.zone_avg)
        return f"{self.method:<14s} k = {self.k:.6f}   zone-avg flux = [{za}]   dofs = {self.ndofs}"


# --------------------------------------------------------------------------- #
def solve_eigen(A: sp.spmatrix, F: sp.spmatrix, tol: float = 1e-12):
    """Fundamental mode of A u = (1/k) F u (largest real k)."""
    lu = spla.splu(sp.csc_matrix(A))
    F = sp.csr_matrix(F)
    n = A.shape[0]
    op = spla.LinearOperator((n, n), matvec=lambda v: lu.solve(F @ v), dtype=float)
    vals, vecs = spla.eigs(op, k=min(3, n - 2), which="LM", tol=tol, v0=np.ones(n))
    i = int(np.argmax(vals.real))
    k = float(vals[i].real)
    u = vecs[:, i].real
    return k, u


def _finish(method, disc: Discretization, k, phi_dofs, info=None) -> Result:
    total = disc.total_integral(phi_dofs)
    phi_dofs = phi_dofs * (disc.problem.length / total)   # slab-average = 1, positive
    x, phi = disc.sample(phi_dofs)
    widths = np.array([z.thickness for z in disc.problem.zones])
    zone_avg = disc.zone_integrals(phi_dofs) / widths
    return Result(method, k, x, phi, zone_avg, info.pop("ndofs") if info else disc.ndofs, info or {})


# --------------------------------------------------------------------------- #
def diffusion(problem: SlabProblem, disc: Discretization | None = None) -> Result:
    d = disc or build(problem)
    A = d.stiffness(lambda z: 1.0 / (3.0 * z.sigma_t)) + d.mass(lambda z: z.sigma_a)
    F = d.mass(lambda z: z.nu_sigma_f)
    k, u = solve_eigen(A, F)
    return _finish("diffusion", d, k, u)


def sp3(problem: SlabProblem, disc: Discretization | None = None) -> Result:
    d = disc or build(problem)
    K0 = d.stiffness(lambda z: 1.0 / (3.0 * z.sigma_t))
    K2 = d.stiffness(lambda z: 9.0 / (35.0 * z.sigma_t))
    Ma = d.mass(lambda z: z.sigma_a)
    Mt = d.mass(lambda z: z.sigma_t)
    Mf = d.mass(lambda z: z.nu_sigma_f)
    A = sp.bmat([[K0 + Ma, -2.0 * Ma],
                 [-0.4 * Ma, K2 + Mt + 0.8 * Ma]], format="csr")
    F = sp.bmat([[Mf, -2.0 * Mf],
                 [-0.4 * Mf, 0.8 * Mf]], format="csr")
    k, u = solve_eigen(A, F)
    n = d.ndofs
    Phi0, phi2 = u[:n], u[n:]
    return _finish("SP3", d, k, Phi0 - 2.0 * phi2,
                   {"ndofs": 2 * n, "phi2_over_phi0_max":
                    float(np.max(np.abs(phi2)) / np.max(np.abs(Phi0 - 2 * phi2)))})


def gauss_legendre_half(N: int):
    if N < 2 or N % 2:
        raise ValueError("S_N order must be even and >= 2")
    mu, w = np.polynomial.legendre.leggauss(N)
    pos = mu > 0
    return mu[pos], w[pos]


def sn_even_parity(problem: SlabProblem, N: int = 16, disc: Discretization | None = None) -> Result:
    d = disc or build(problem)
    mu, w = gauss_legendre_half(N)
    Kt = d.stiffness(lambda z: 1.0 / z.sigma_t)
    Mt = d.mass(lambda z: z.sigma_t)
    Ms = d.mass(lambda z: z.sigma_s)
    Mf = d.mass(lambda z: z.nu_sigma_f)
    # Rows are multiplied by w_m, which makes both A and F symmetric:
    #   A_mn = w_m (mu_m^2 Kt + Mt) delta_mn - w_m w_n Ms
    #   F_mn = w_m w_n Mf
    ww = np.outer(w, w)
    A = sp.bmat([[(w[m] * (mu[m] ** 2 * Kt + Mt) if m == n_ else None) for n_ in range(len(mu))]
                 for m in range(len(mu))], format="csr") - sp.kron(ww, Ms, format="csr")
    F = sp.kron(ww, Mf, format="csr")
    k, u = solve_eigen(A, F)
    n = d.ndofs
    psi = u.reshape(len(mu), n)
    phi = 2.0 * (w @ psi)
    return _finish(f"S{N} (even-par)", d, k, phi, {"ndofs": len(mu) * n})


def solve_all(problem: SlabProblem, sn_orders=(16,)) -> list[Result]:
    d = build(problem)
    out = [diffusion(problem, d), sp3(problem, d)]
    out += [sn_even_parity(problem, N, d) for N in sn_orders]
    return out
