r"""Multigroup (G-group) k-eigenvalue solvers for a reflected multi-zone slab.

Same three angular models as the one-group code -- P1 (= diffusion), SP3 and
even-parity S_N -- on the same nutils mesh and per-zone matrices K_z, M_z.
The unknowns of all groups are solved together as one generalized eigenproblem

    A u = (1/k) F u,

so there are no inner/outer iterations: group coupling (down- and up-scatter,
fission spectrum) sits in the off-diagonal blocks of A and F.

Data per zone (G groups):
    sigma_c[g], sigma_f[g], nu[g], chi[g]
    S_l[g, g'] = Σs,l(g -> g'),  l = 0..L   (row = from-group, column = to-group)
    Σt,g = sigma_c[g] + sigma_f[g] + sum_g' S_0[g, g']

The scattering *into* group g from moment φ_l is (S_l^T φ_l)_g, so the
moment removal matrices are

    R_l = diag(Σt) - S_l^T        (G x G),   R_0 = absorption + net out-scatter.

P1 / diffusion
--------------
    J = -(1/3) R_1^{-1} φ'           (G x G diffusion "tensor" D = R_1^{-1}/3)
    -(D φ')' + R_0 φ = (1/k) χ (νΣf)^T φ

With S_1 diagonal this is ordinary multigroup diffusion with D_g = 1/(3(Σt,g - Σs1,g->g)).

SP3 (= P3 in slab geometry), Φ0 = φ0 + 2 φ2, scalar flux φ = Φ0 - 2φ2
-------------------------------------------------------------------
    -(1/3 (R_1^{-1} Φ0')' + R_0 (Φ0 - 2φ2)                       =  (1/k) χ νΣf^T (Φ0 - 2φ2)
    -(9/35)(R_3^{-1} φ2')' + R_2 φ2 - (2/5) R_0 (Φ0 - 2φ2)      = -(2/5)(1/k) χ νΣf^T (Φ0 - 2φ2)

Even-parity S_N, combined index a = (g, m) over groups and directions μ_m > 0
------------------------------------------------------------------------------
    S_par[(g,m), (g',n)] = Σ_{l of parity} (2l+1) Σs,l(g' -> g) P_l(μ_m) P_l(μ_n) w_n
    T = diag(Σt,g) ⊗ I - S_odd,      ψ⁻ = -T^{-1} μ ψ⁺'
    -(μ T^{-1} μ ψ⁺')' + (diag(Σt) - S_even) ψ⁺ = (1/2k) χ_g Σ_g' νΣf,g' φ_g',   φ_g = Σ_m 2 w_m ψ⁺_{g,m}

Rows are multiplied by w_m (as in the one-group code). For G = 1 every model
reduces exactly to the one-group solvers in `solvers.py`.
"""
from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numpy.polynomial import legendre as npleg

from .fem import Discretization, build
from .solvers import gauss_legendre_half, solve_eigen


# --------------------------------------------------------------------------- #
# problem definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MGZone:
    thickness: float
    sigma_c: tuple
    sigma_f: tuple
    scatter: tuple                      # (L+1, G, G): scatter[l][g][g'] = Σs,l(g -> g')
    nu: tuple
    chi: tuple
    name: str = ""

    def __post_init__(self):
        S = np.asarray(self.scatter, float)
        if S.ndim == 2:
            S = S[None]
        G = S.shape[1]
        object.__setattr__(self, "scatter", S)
        for k in ("sigma_c", "sigma_f", "nu", "chi"):
            v = np.asarray(getattr(self, k), float).reshape(-1)
            if v.size == 1:
                v = np.full(G, v[0])
            if v.size != G:
                raise ValueError(f"zone {self.name!r}: {k} has {v.size} entries, expected {G}")
            object.__setattr__(self, k, v)
        if S.shape[1:] != (G, G):
            raise ValueError(f"zone {self.name!r}: scattering matrices must be {G}x{G}")
        if self.thickness <= 0:
            raise ValueError(f"zone {self.name!r}: thickness must be > 0")
        if np.any(S[0] < 0) or np.any(self.sigma_c < 0) or np.any(self.sigma_f < 0) or np.any(self.nu < 0):
            raise ValueError(f"zone {self.name!r}: P0 cross sections must be >= 0")
        for l in range(1, S.shape[0]):
            if np.any(np.abs(S[l]) > S[0] * (1 + 1e-12) + 1e-300):
                raise ValueError(f"zone {self.name!r}: |Σs{l}(g->g')| exceeds Σs0(g->g') for some transfer")
        if np.any(self.sigma_t_g <= 0):
            raise ValueError(f"zone {self.name!r}: total cross section must be > 0 in every group")
        if self.L > 0:
            mu0 = np.linspace(-1, 1, 2001)
            bad = []
            for g in range(G):
                for h in range(G):
                    if S[0, g, h] > 0:
                        c = [(2 * l + 1) / 2 * S[l, g, h] for l in range(self.L + 1)]
                        if npleg.legval(mu0, c).min() < -1e-12 * S[0, g, h]:
                            bad.append(f"{g + 1}->{h + 1}")
            if bad:
                warnings.warn(f"zone {self.name!r}: truncated P{self.L} scattering kernel is negative for some "
                              f"angles in transfer(s) {', '.join(bad)}; results stay well-defined", stacklevel=3)

    @property
    def G(self) -> int:
        return self.scatter.shape[1]

    @property
    def L(self) -> int:
        return self.scatter.shape[0] - 1

    def S(self, l: int) -> np.ndarray:
        return self.scatter[l] if l <= self.L else np.zeros((self.G, self.G))

    @property
    def sigma_t_g(self) -> np.ndarray:
        return self.sigma_c + self.sigma_f + self.scatter[0].sum(axis=1)

    @property
    def sigma_a_g(self) -> np.ndarray:
        return self.sigma_c + self.sigma_f

    @property
    def nu_sigma_f(self) -> np.ndarray:
        return self.nu * self.sigma_f

    @property
    def sigma_t(self) -> float:          # used by fem.build for the mesh density
        return float(self.sigma_t_g.max())

    def R(self, l: int) -> np.ndarray:
        """Moment removal matrix R_l = diag(Σt) - S_l^T."""
        return np.diag(self.sigma_t_g) - self.S(l).T

    def fission_matrix(self) -> np.ndarray:
        """χ (νΣf)^T, (G x G)."""
        return np.outer(self.chi, self.nu_sigma_f)


@dataclass
class MGProblem:
    zones: list[MGZone]
    elements_per_mfp: float = 4.0
    min_elements_per_zone: int = 8
    degree: int = 2
    title: str = "slab"
    group_names: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.zones:
            raise ValueError("need at least one zone")
        G = {z.G for z in self.zones}
        if len(G) != 1:
            raise ValueError("all zones must have the same number of groups")
        if not any(np.any(z.nu_sigma_f > 0) for z in self.zones):
            raise ValueError("no fissile zone: k-eigenvalue problem is undefined")
        if not self.group_names:
            self.group_names = [f"group {g + 1}" for g in range(self.G)]

    @property
    def G(self) -> int:
        return self.zones[0].G

    @property
    def length(self) -> float:
        return sum(z.thickness for z in self.zones)

    @property
    def interfaces(self) -> list[float]:
        x, out = 0.0, [0.0]
        for z in self.zones:
            x += z.thickness
            out.append(x)
        return out

    @classmethod
    def from_toml(cls, path) -> "MGProblem":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        G = int(data.pop("groups"))
        chi_default = data.pop("chi", [1.0] + [0.0] * (G - 1))
        names = data.pop("group_names", [])
        zones = []
        for z in data.pop("zone"):
            S0 = np.asarray(z.pop("sigma_s"), float).reshape(G, G)
            Sl = [np.asarray(m, float).reshape(G, G) for m in z.pop("sigma_s_l", [])]
            zones.append(MGZone(thickness=z.pop("thickness"),
                                sigma_c=z.pop("sigma_c"),
                                sigma_f=z.pop("sigma_f", [0.0] * G),
                                scatter=np.stack([S0] + Sl),
                                nu=z.pop("nu", [2.43] * G),
                                chi=z.pop("chi", chi_default),
                                name=z.pop("name", "")))
            if z:
                raise ValueError(f"unknown zone keys: {sorted(z)}")
        mesh = data.pop("mesh", {})
        return cls(zones=zones, title=data.pop("title", Path(path).stem), group_names=names, extra=data, **mesh)


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class MGResult:
    method: str
    k: float
    x: np.ndarray
    phi: np.ndarray            # (G, npts) scalar flux per group
    zone_avg: np.ndarray       # (G, nzones)
    ndofs: int
    info: dict = field(default_factory=dict)

    def normalized(self, problem: MGProblem, mode: str = "average") -> "MGResult":
        """average: slab- and group-summed average flux = 1; fission: total fission rate = 1."""
        t = np.array([z.thickness for z in problem.zones])
        if mode == "average":
            c = problem.length / float((self.zone_avg * t).sum())
        elif mode == "fission":
            sf = np.array([z.sigma_f for z in problem.zones]).T          # (G, Z)
            c = 1.0 / float((self.zone_avg * sf * t).sum())
        else:
            raise ValueError(f"unknown normalisation {mode!r}")
        return MGResult(self.method, self.k, self.x, c * self.phi, c * self.zone_avg, self.ndofs,
                        {**self.info, "norm": mode})

    def summary(self) -> str:
        za = "; ".join("[" + ", ".join(f"{v:.5g}" for v in row) + "]" for row in self.zone_avg)
        return f"{self.method:<14s} k = {self.k:.6f}   zone-avg flux per group = {za}"


def _finish(method, problem, d: Discretization, k, phis, ndofs) -> MGResult:
    phis = np.asarray(phis)                         # (G, n)
    if phis.sum() < 0:
        phis = -phis
    t = np.array([z.thickness for z in problem.zones])
    za = np.array([d.zone_integrals(p) for p in phis]) / t
    ph = []
    for p in phis:
        x, f = d.sample(p)
        ph.append(f)
    r = MGResult(method, k, x, np.array(ph), za, ndofs)
    return r.normalized(problem, "average")


def _kron_zones(problem, d, coefK, coefM):
    """sum_z kron(coefK(z), K_z) + kron(coefM(z), M_z); coef* return dense matrices or None."""
    A = None
    for z, Kz, Mz in zip(problem.zones, d.K, d.M):
        for cf, Z in ((coefK, Kz), (coefM, Mz)):
            C = cf(z)
            if C is None:
                continue
            blk = sp.kron(sp.csr_matrix(C), Z, format="csr")
            A = blk if A is None else A + blk
    return A


# --------------------------------------------------------------------------- #
# solvers
# --------------------------------------------------------------------------- #
def p1(problem: MGProblem, d: Discretization | None = None) -> MGResult:
    d = d or build(problem)
    A = _kron_zones(problem, d, lambda z: np.linalg.inv(z.R(1)) / 3.0, lambda z: z.R(0))
    F = _kron_zones(problem, d, lambda z: None, lambda z: z.fission_matrix())
    k, u = solve_eigen(A, F)
    n, G = d.ndofs, problem.G
    return _finish("P1 (diffusion)", problem, d, k, u.reshape(G, n), G * n)


def sp3(problem: MGProblem, d: Discretization | None = None) -> MGResult:
    d = d or build(problem)
    G, n = problem.G, d.ndofs
    # unknown ordering: [Φ0 (G blocks), φ2 (G blocks)]
    def K_(z):
        Z = np.zeros((2 * G, 2 * G))
        Z[:G, :G] = np.linalg.inv(z.R(1)) / 3.0
        Z[G:, G:] = 9.0 / 35.0 * np.linalg.inv(z.R(3))
        return Z

    def M_(z):
        R0, R2 = z.R(0), z.R(2)
        Z = np.zeros((2 * G, 2 * G))
        Z[:G, :G] = R0
        Z[:G, G:] = -2.0 * R0
        Z[G:, :G] = -0.4 * R0
        Z[G:, G:] = R2 + 0.8 * R0
        return Z

    def F_(z):
        X = z.fission_matrix()
        Z = np.zeros((2 * G, 2 * G))
        Z[:G, :G], Z[:G, G:] = X, -2.0 * X
        Z[G:, :G], Z[G:, G:] = -0.4 * X, 0.8 * X
        return Z

    A = _kron_zones(problem, d, K_, M_)
    F = _kron_zones(problem, d, lambda z: None, F_)
    k, u = solve_eigen(A, F)
    u = u.reshape(2 * G, n)
    return _finish("SP3", problem, d, k, u[:G] - 2.0 * u[G:], 2 * G * n)


def _sn_blocks(z: MGZone, mu, w):
    """Angular-group matrices of one zone: stiffness C, removal R, both (G*Nh)^2, rows scaled by w."""
    G, Nh = z.G, len(mu)
    N = 2 * Nh
    L = min(z.L, N - 1)
    P = np.array([npleg.Legendre.basis(l)(mu) for l in range(L + 1)])      # (L+1, Nh)
    St = np.kron(np.diag(z.sigma_t_g), np.eye(Nh))
    Sodd = np.zeros((G * Nh, G * Nh))
    Seven = np.zeros((G * Nh, G * Nh))
    for l in range(L + 1):
        ang = (2 * l + 1) * np.outer(P[l], P[l] * w)                       # [m, n] = (2l+1) P_l(μm) P_l(μn) w_n
        blk = np.kron(z.S(l).T, ang)                                       # [(g,m),(g',n)] uses Σs,l(g'->g)
        if l % 2:
            Sodd += blk
        else:
            Seven += blk
    Wf = np.kron(np.eye(G), np.diag(w))
    Mu = np.kron(np.eye(G), np.diag(mu))
    T = St - Sodd
    C = Wf @ Mu @ np.linalg.solve(T, Mu)
    R = Wf @ (St - Seven)
    return C, R


def sn(problem: MGProblem, N: int = 16, d: Discretization | None = None) -> MGResult:
    d = d or build(problem)
    mu, w = gauss_legendre_half(N)
    lost = [z.name or str(i) for i, z in enumerate(problem.zones, 1) if z.L > N - 1]
    if lost:
        warnings.warn(f"S{N}: scattering moments above P{N - 1} dropped in zone(s) {', '.join(lost)}", stacklevel=2)
    G, Nh, n = problem.G, len(mu), d.ndofs
    blocks = [_sn_blocks(z, mu, w) for z in problem.zones]
    A = None
    for (C, R), Kz, Mz in zip(blocks, d.K, d.M):
        blk = sp.kron(sp.csr_matrix(C), Kz, format="csr") + sp.kron(sp.csr_matrix(R), Mz, format="csr")
        A = blk if A is None else A + blk
    ww = np.outer(w, w)
    F = _kron_zones(problem, d, lambda z: None, lambda z: np.kron(z.fission_matrix(), ww))
    k, u = solve_eigen(A, F)
    psi = u.reshape(G, Nh, n)
    phis = 2.0 * np.einsum("m,gmi->gi", w, psi)
    return _finish(f"S{N} (even-par)", problem, d, k, phis, G * Nh * n)


def solve_all(problem: MGProblem, sn_orders=(16,), norm: str = "average") -> list[MGResult]:
    d = build(problem)
    out = [p1(problem, d), sp3(problem, d)] + [sn(problem, N, d) for N in sn_orders]
    return [r.normalized(problem, norm) for r in out]


def k_infinity(zone: MGZone) -> float:
    """Infinite-medium multiplication factor of one zone: largest eigenvalue of R_0^{-1} χ νΣf^T."""
    ev = np.linalg.eigvals(np.linalg.solve(zone.R(0), zone.fission_matrix()))
    return float(np.max(ev.real))
