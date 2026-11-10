r"""One-group k-eigenvalue solvers for a reflected multi-zone slab.

All three models are written as a generalized eigenproblem

    A u = (1/k) F u

and solved for the fundamental mode (largest k) with ARPACK on A^{-1} F.

Notation per zone: St = total, Ss_l = l-th Legendre moment of scattering
(Ss_0 = Ss), s_n = St - Ss_n, Sa = s_0 = absorption, nSf = nu * fission.
Scattering may be anisotropic, Ss(mu0) = sum_l (2l+1)/2 Ss_l P_l(mu0);
fission is isotropic.

Diffusion (P1)
--------------
    -(D phi')' + Sa phi = (1/k) nSf phi,          D = 1/(3 s_1)

(transport-corrected diffusion coefficient; moments above l = 1 do not enter).

SP3 (Gelbard; Brantley & Larsen form)
-------------------------------------
Unknowns Phi0 = phi0 + 2 phi2 and phi2; scalar flux phi = Phi0 - 2 phi2.

    -(1/(3s_1) Phi0')' + Sa (Phi0 - 2phi2)                  = (1/k) nSf (Phi0 - 2phi2)
    -(9/(35s_3) phi2')' + s_2 phi2 - 2/5 Sa (Phi0 - 2phi2)  = -2/5 (1/k) nSf (Phi0 - 2phi2)

In 1D planar geometry SP3 is identical to P3 (scattering moments up to l = 3
are used), so this is a genuine angular refinement of diffusion. Reflective
BCs (phi1 = phi3 = 0) and P3 interface conditions are both natural in this
weak form. With isotropic scattering s_1 = s_2 = s_3 = St.

Transport (even-parity S_N)
---------------------------
Gauss-Legendre directions mu_m > 0, full-range weights w_m (sum over all = 2),
psi+- = [psi(mu) +- psi(-mu)] / 2. Even moments of the scattering source
involve only psi+, odd moments only psi-:

    odd :  mu psi+' + T psi-   = 0,        T  = St I - S_odd
    even:  mu psi-' + St psi+  = S_even psi+ + (nSf/2k) phi

    S_par[m, n] = sum_(l of that parity) (2l+1) Ss_l P_l(mu_m) P_l(mu_n) w_n

Eliminating psi- = -T^-1 mu psi+' gives N/2 coupled second-order equations

    -(mu T^-1 mu psi+')' + (St I - S_even) psi+ = (nSf/2k) phi,   phi = sum_m 2 w_m psi+_m

which are made symmetric by multiplying the rows by W = diag(w):
stiffness (W mu)(W T)^-1(W mu), removal W(St I - S_even), fission w w^T nSf.
With isotropic scattering T = St I and the stiffness is diagonal, mu^2/St.
Moments with l > N - 1 cannot be resolved by an N-point quadrature and are
dropped (with a warning).

Reflection psi(mu) = psi(-mu) means zero odd parity, i.e. a natural BC. As N
grows this converges to the exact transport solution; N = 16-32 is the
reference in the comparison. An independent diamond-difference S_N code in
`reference.py` is used in the tests to cross-check it.
"""
from __future__ import annotations

import warnings
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
    A = d.stiffness(lambda z: 1.0 / (3.0 * z.sigma_n(1))) + d.mass(lambda z: z.sigma_a)
    F = d.mass(lambda z: z.nu_sigma_f)
    k, u = solve_eigen(A, F)
    return _finish("diffusion", d, k, u)


def sp3(problem: SlabProblem, disc: Discretization | None = None) -> Result:
    d = disc or build(problem)
    K0 = d.stiffness(lambda z: 1.0 / (3.0 * z.sigma_n(1)))
    K2 = d.stiffness(lambda z: 9.0 / (35.0 * z.sigma_n(3)))
    Ma = d.mass(lambda z: z.sigma_a)
    M2 = d.mass(lambda z: z.sigma_n(2))
    Mf = d.mass(lambda z: z.nu_sigma_f)
    A = sp.bmat([[K0 + Ma, -2.0 * Ma],
                 [-0.4 * Ma, K2 + M2 + 0.8 * Ma]], format="csr")
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


def sn_angular_blocks(zone, mu, w):
    """Symmetric angular matrices of one zone for the even-parity S_N system.

    Returns (C, R): stiffness-coefficient matrix (W mu)(W T)^-1(W mu) and removal
    matrix W (St I - S_even), both (N/2 x N/2). Fission is w w^T nSf.
    """
    N = 2 * len(mu)
    L = min(zone.scattering_order, N - 1)
    W = np.diag(w)
    Todd = zone.sigma_t * W
    R = zone.sigma_t * W
    for l in range(L + 1):
        p = w * np.polynomial.legendre.Legendre.basis(l)(mu)       # w_m P_l(mu_m)
        term = (2 * l + 1) * zone.sigma_sl(l) * np.outer(p, p)
        if l % 2:
            Todd = Todd - term
        else:
            R = R - term
    Wmu = np.diag(w * mu)
    C = Wmu @ np.linalg.solve(Todd, Wmu)
    return 0.5 * (C + C.T), R


def sn_even_parity(problem: SlabProblem, N: int = 16, disc: Discretization | None = None) -> Result:
    d = disc or build(problem)
    mu, w = gauss_legendre_half(N)
    lost = [z.name or str(i) for i, z in enumerate(problem.zones, 1) if z.scattering_order > N - 1]
    if lost:
        warnings.warn(f"S{N}: scattering moments above P{N - 1} dropped in zone(s) {', '.join(lost)}",
                      stacklevel=2)
    # A = sum_z kron(C_z, K_z) + kron(R_z, M_z);   F = sum_z kron(w w^T, nSf_z M_z)
    A = None
    for z, Kz, Mz in zip(problem.zones, d.K, d.M):
        C, R = sn_angular_blocks(z, mu, w)
        blk = sp.kron(C, Kz, format="csr") + sp.kron(R, Mz, format="csr")
        A = blk if A is None else A + blk
    F = sp.kron(np.outer(w, w), d.mass(lambda z: z.nu_sigma_f), format="csr")
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
