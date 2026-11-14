r"""Multigroup discrete-ordinates (S_N) transport in 1D cylindrical geometry.

Infinite cylinder, radial coordinate r in [0, R], symmetry axis at r = 0.
Conservative form of the transport equation (Lewis & Miller):

    μ/r ∂(r ψ)/∂r - (1/r) ∂(η ψ)/∂ω + Σt ψ = q,      μ = sinθ cosω,  η = sinθ sinω

with ξ = cosθ the cosine to the cylinder axis. Discretisation:

* angles: product quadrature. Polar levels ξ_p, w_p = Gauss-Legendre on (0, 1]
  (up/down symmetry), azimuth ω_m = equal-weight midpoints on (0, π) (η symmetry),
  ordered by increasing μ. Direction weight W_pm = 4 w_p π/N_a, Σ W = 4π.
* angular redistribution: α_{m+1/2} = α_{m-1/2} - w_m μ_m, α_{1/2} = 0 (with w_m = π/N_a),
  which keeps a spatially flat, isotropic flux exactly flat; the "starting direction"
  μ = -sinθ (ω = π) gives the first angular edge flux on every level.
* space and angle: diamond difference; per unit height and radian A = r, V = Δ(r²)/2.
* scattering: Legendre P_L in Ω·Ω' truncated at L = 1 (P1):
      q_g(μ) = 1/(4π) [ Σ_g' Σs0(g'->g) φ_g' + 3 μ Σ_g' Σs1(g'->g) J_g' + χ_g Σ_g' νΣf,g' φ_g' / k ]
  with J the radial current (higher moments are not supported here).
* boundaries: r = 0 symmetry (ψ(μ) = ψ(-μ)); r = R white (isotropic return of the
  outgoing partial current), i.e. zero net current like the reflective/natural
  condition of the P1 and SP3 solvers.

Source iteration and power iteration run in one loop. The sweep is compiled with
numba when it is installed (pip install numba), otherwise it runs in plain Python.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:                               # pragma: no cover - fallback
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap(a[0]) if a and callable(a[0]) else wrap


@njit(cache=True)
def _sweep(re, st, q0, q1, xi_w, s_p, mu, wa, alpha, psi_in):
    """One transport sweep for all groups.

    re: (nc+1,) edges; st: (G, nc); q0, q1: (G, nc) isotropic and P1 source moments
    (q(μ) = (q0 + 3 μ q1)/(4π)); mu: (Np, Na); alpha: (Np, Na+1); psi_in: (G,) white return.
    Returns phi (G, nc), J (G, nc), J_out (G,), number of negative cell fluxes.
    """
    G, nc = st.shape
    Np, Na = mu.shape
    phi = np.zeros((G, nc))
    cur = np.zeros((G, nc))
    jout = np.zeros(G)
    nneg = 0
    four_pi = 4.0 * np.pi
    for g in range(G):
        for p in range(Np):
            s = s_p[p]
            ang = np.zeros(nc)                    # angular edge flux ψ_{m-1/2, i}
            # ---- starting direction μ = -s (ω = π): -s dψ/dr + Σt ψ = q
            edge = psi_in[g]
            for i in range(nc - 1, -1, -1):
                dr = re[i + 1] - re[i]
                qq = (q0[g, i] - 3.0 * s * q1[g, i]) / four_pi
                c = (qq + 2.0 * s / dr * edge) / (2.0 * s / dr + st[g, i])
                ang[i] = c
                edge = 2.0 * c - edge
            center = np.zeros(Na)                 # ψ at r = 0 for each direction (μ < 0 ones)
            for m in range(Na):
                mm = mu[p, m]
                W = xi_w[p] * wa[m] * 4.0
                ap, am = alpha[p, m + 1], alpha[p, m]
                if mm < 0.0:                      # inward sweep from the outer boundary
                    edge = psi_in[g]
                    for i in range(nc - 1, -1, -1):
                        Ap, Am = re[i + 1], re[i]
                        dA = Ap - Am
                        V = 0.5 * (Ap * Ap - Am * Am)
                        qq = (q0[g, i] + 3.0 * mm * q1[g, i]) / four_pi
                        den = -2.0 * mm * Am + 2.0 * dA * ap / wa[m] + st[g, i] * V
                        c = (V * qq - mm * (Ap + Am) * edge + dA / wa[m] * (ap + am) * ang[i]) / den
                        if c < 0.0:
                            nneg += 1
                        phi[g, i] += W * c
                        cur[g, i] += W * mm * c
                        ang[i] = 2.0 * c - ang[i]
                        edge = 2.0 * c - edge
                    center[m] = edge
                else:                             # outward sweep from the axis
                    edge = center[Na - 1 - m]     # symmetry: ψ(0, μ) = ψ(0, -μ)
                    for i in range(nc):
                        Ap, Am = re[i + 1], re[i]
                        dA = Ap - Am
                        V = 0.5 * (Ap * Ap - Am * Am)
                        qq = (q0[g, i] + 3.0 * mm * q1[g, i]) / four_pi
                        den = 2.0 * mm * Ap + 2.0 * dA * ap / wa[m] + st[g, i] * V
                        c = (V * qq + mm * (Ap + Am) * edge + dA / wa[m] * (ap + am) * ang[i]) / den
                        if c < 0.0:
                            nneg += 1
                        phi[g, i] += W * c
                        cur[g, i] += W * mm * c
                        ang[i] = 2.0 * c - ang[i]
                        edge = 2.0 * c - edge
                    jout[g] += W * mm * edge
    return phi, cur, jout, nneg


def quadrature(n_polar: int, n_azimuth: int):
    """Product quadrature: returns xi_w (Np,), s_p (Np,), mu (Np, Na), wa (Na,), alpha (Np, Na+1)."""
    x, w = np.polynomial.legendre.leggauss(2 * n_polar)
    keep = x > 0
    xi, xi_w = x[keep], w[keep]                   # Σ xi_w = 1
    s_p = np.sqrt(1.0 - xi ** 2)
    omega = np.pi - (np.arange(n_azimuth) + 0.5) * np.pi / n_azimuth   # π..0, μ increasing
    wa = np.full(n_azimuth, np.pi / n_azimuth)
    mu = s_p[:, None] * np.cos(omega)[None, :]
    alpha = np.zeros((n_polar, n_azimuth + 1))
    for m in range(n_azimuth):
        alpha[:, m + 1] = alpha[:, m] - wa[m] * mu[:, m]
    alpha[:, -1] = 0.0                            # exact by symmetry; remove round-off
    return xi_w, s_p, mu, wa, alpha


def solve(problem, n_polar: int = 8, n_azimuth: int = 16, cells_per_mfp: float = 4.0,
          min_cells_per_zone: int = 40, tol: float = 1e-9, max_iter: int = 200000):
    """k-eigenvalue in a cylinder (MGProblem with geometry='cylinder').

    Returns dict with k, r (cell centres), phi (G, nc), J (G, nc), cell volumes, iterations.
    """
    G = problem.G
    if any(z.L > 1 for z in problem.zones):
        raise NotImplementedError("cylindrical S_N supports scattering up to P1")
    re, zid = [0.0], []
    for iz, z in enumerate(problem.zones):
        nc = max(min_cells_per_zone, int(np.ceil(z.thickness * z.sigma_t * cells_per_mfp)))
        re += list(np.linspace(re[-1], re[-1] + z.thickness, nc + 1)[1:])
        zid += [iz] * nc
    re, zid = np.asarray(re), np.asarray(zid)
    nc = len(zid)
    vol = 0.5 * (re[1:] ** 2 - re[:-1] ** 2)
    zones = problem.zones
    st = np.array([zones[i].sigma_t_g for i in zid]).T.copy()
    nsf = np.array([zones[i].nu_sigma_f for i in zid]).T
    chi = np.array([zones[i].chi for i in zid]).T
    S0 = np.array([zones[i].S(0).T for i in zid])            # (nc, G_to, G_from)
    S1 = np.array([zones[i].S(1).T for i in zid])
    xi_w, s_p, mu, wa, alpha = quadrature(n_polar, n_azimuth)
    w_in = np.sum(4.0 * xi_w[:, None] * wa[None, :] * np.abs(np.minimum(mu, 0.0)))   # Σ_{μ<0} W|μ|

    phi = np.ones((G, nc))
    cur = np.zeros((G, nc))
    k = 1.0
    psi_in = np.zeros(G)
    for it in range(max_iter):
        fis = chi * (nsf * phi).sum(axis=0)[None, :] / k
        q0 = np.einsum("igh,hi->gi", S0, phi) + fis
        q1 = np.einsum("igh,hi->gi", S1, cur)
        new, newcur, jout, nneg = _sweep(re, st, q0, q1, xi_w, s_p, mu, wa, alpha, psi_in)
        psi_in = jout / w_in
        prod_old = (nsf * phi * vol).sum()
        prod_new = (nsf * new * vol).sum()
        k_new = k * prod_new / prod_old
        err = max(abs(k_new - k), np.abs(new / new.max() - phi / phi.max()).max())
        phi, cur, k = new, newcur, k_new
        if err < tol and it > 5:
            break
    else:
        raise RuntimeError("cylindrical S_N did not converge")
    return dict(k=k, r=0.5 * (re[1:] + re[:-1]), edges=re, phi=phi, J=cur, vol=vol, zid=zid,
                iterations=it + 1, negative_cells=int(nneg),
                quadrature=f"{n_polar} polar x {2 * n_azimuth} azimuthal (product)")
