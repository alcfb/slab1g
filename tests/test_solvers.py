from pathlib import Path

import numpy as np
import pytest

from slab1g import SlabProblem, Zone, diffusion, sn_even_parity, solve_all, sp3
from slab1g.fem import build
from slab1g.reference import dd_sn

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def cell():
    return SlabProblem.from_toml(EXAMPLES / "fuel_moderator.toml")


def test_infinite_medium_limit():
    """Two identical zones with reflective BCs = infinite medium: k = k_inf, flat flux."""
    z = Zone(thickness=1.0, sigma_s=0.6, sigma_c=0.1, sigma_f=0.08, nu=2.4)
    p = SlabProblem([z, z])
    kinf = z.nu_sigma_f / z.sigma_a
    for r in solve_all(p, (8,)):
        assert r.k == pytest.approx(kinf, rel=1e-10), r.method
        np.testing.assert_allclose(r.phi, 1.0, atol=1e-8)


def test_s2_is_diffusion_and_s4_is_sp3(cell):
    """Slab geometry with Gauss quadrature: S_N == P_(N-1); P1 == diffusion, P3 == SP3."""
    d = build(cell)
    rd, r3 = diffusion(cell, d), sp3(cell, d)
    s2, s4 = sn_even_parity(cell, 2, d), sn_even_parity(cell, 4, d)
    assert s2.k == pytest.approx(rd.k, rel=1e-10)
    assert s4.k == pytest.approx(r3.k, rel=1e-10)
    np.testing.assert_allclose(s2.phi, rd.phi, rtol=1e-8)
    np.testing.assert_allclose(s4.phi, r3.phi, rtol=1e-8)


@pytest.mark.parametrize("method", ["diffusion", "sp3", "s8"])
def test_neutron_balance(cell, method):
    """Reflective on both sides means no leakage: k = int nuSf phi / int Sa phi."""
    d = build(cell)
    r = {"diffusion": lambda: diffusion(cell, d), "sp3": lambda: sp3(cell, d),
         "s8": lambda: sn_even_parity(cell, 8, d)}[method]()
    t = np.array([z.thickness for z in cell.zones])
    prod = sum(z.nu_sigma_f * a * w for z, a, w in zip(cell.zones, r.zone_avg, t))
    absn = sum(z.sigma_a * a * w for z, a, w in zip(cell.zones, r.zone_avg, t))
    assert r.k == pytest.approx(prod / absn, rel=1e-9)
    assert np.dot(r.zone_avg, t) / cell.length == pytest.approx(1.0, rel=1e-10)


def test_even_parity_matches_independent_dd_sn(cell):
    """nutils even-parity S16 vs a plain diamond-difference S16 sweep code."""
    r = sn_even_parity(cell, 16)
    k_dd, x, phi_dd, _ = dd_sn(cell, 16, cells_per_mfp=80)
    assert r.k == pytest.approx(k_dd, abs=2e-6)
    phi_fe = np.interp(x, r.x, r.phi)
    np.testing.assert_allclose(phi_fe, phi_dd, rtol=2e-4)


def test_angular_convergence_ordering(cell):
    res = solve_all(cell, (4, 8, 16, 32))
    k = {r.method.split()[0]: r.k for r in res}
    ref = k["S32"]
    errs = [abs(k[m] - ref) for m in ("diffusion", "SP3", "S8", "S16")]
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] < 2e-5


def test_mesh_convergence(cell):
    fine = SlabProblem(cell.zones, elements_per_mfp=40, min_elements_per_zone=64)
    assert diffusion(cell).k == pytest.approx(diffusion(fine).k, abs=1e-8)
    assert sn_even_parity(cell, 8).k == pytest.approx(sn_even_parity(fine, 8).k, abs=1e-7)


def test_input_validation():
    with pytest.raises(ValueError):
        Zone(thickness=-1, sigma_s=1, sigma_c=0)
    with pytest.raises(ValueError):
        SlabProblem([Zone(thickness=1, sigma_s=1, sigma_c=0.1)])  # nothing fissile
    with pytest.raises(ValueError):
        sn_even_parity(SlabProblem.from_toml(EXAMPLES / "fuel_moderator.toml"), 5)


def test_cli_runs(tmp_path):
    from slab1g.cli import main
    assert main([str(EXAMPLES / "fuel_moderator.toml"), "--sn", "4", "8",
                 "--plot", str(tmp_path / "f.png"), "--csv", str(tmp_path / "f.csv")]) == 0
    assert (tmp_path / "f.png").stat().st_size > 0
    assert (tmp_path / "f.csv").read_text().startswith("x,diffusion,SP3")


# --------------------------------------------------------------------------- #
# Anisotropic scattering
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def aniso():
    return SlabProblem.from_toml(EXAMPLES / "anisotropic_cell.toml")


def test_isotropic_input_unchanged():
    z = Zone(thickness=1.0, sigma_s=0.6, sigma_c=0.1, sigma_f=0.08)
    assert z.scattering_order == 0 and z.sigma_n(1) == z.sigma_t == z.sigma_n(3)
    zl = Zone(thickness=1.0, sigma_s=0.6, sigma_c=0.1, sigma_f=0.08, sigma_s_l=(0.0, 0.0, 0.0))
    p, pl = SlabProblem([z, z]), SlabProblem([zl, zl])
    assert sn_even_parity(p, 8).k == pytest.approx(sn_even_parity(pl, 8).k, rel=1e-12)


def test_kernel_normalisation(aniso):
    from numpy.polynomial.legendre import leggauss, Legendre
    x, w = leggauss(40)
    for z in aniso.zones:
        f = z.scattering_kernel(x)
        assert w @ f == pytest.approx(z.sigma_s, rel=1e-12)
        assert w @ (f * x) == pytest.approx(z.sigma_sl(1), rel=1e-12)   # Σs1 = Σs0 · mean cosine
        assert w @ (f * Legendre.basis(2)(x)) == pytest.approx(z.sigma_sl(2), abs=1e-12)


def test_anisotropic_infinite_medium():
    """Anisotropy cannot change k_inf or the flat flux of an infinite medium."""
    z = Zone(thickness=1.0, sigma_s=0.6, sigma_c=0.1, sigma_f=0.08, sigma_s_l=(0.3, 0.1, 0.02))
    p = SlabProblem([z, z])
    for r in solve_all(p, (8,)):
        assert r.k == pytest.approx(z.nu_sigma_f / z.sigma_a, rel=1e-10), r.method
        np.testing.assert_allclose(r.phi, 1.0, atol=1e-8)


def test_anisotropic_s2_is_transport_corrected_diffusion_and_s4_is_sp3(aniso):
    d = build(aniso)
    rd, r3 = diffusion(aniso, d), sp3(aniso, d)
    with pytest.warns(UserWarning, match="dropped"):
        s2 = sn_even_parity(aniso, 2, d)      # Σs2, Σs3 cannot be represented by S2
    s4 = sn_even_parity(aniso, 4, d)
    assert s2.k == pytest.approx(rd.k, rel=1e-10)
    assert s4.k == pytest.approx(r3.k, rel=1e-10)
    np.testing.assert_allclose(s2.phi, rd.phi, rtol=1e-8)
    np.testing.assert_allclose(s4.phi, r3.phi, rtol=1e-8)


def test_anisotropic_matches_independent_dd_sn(aniso):
    r = sn_even_parity(aniso, 16)
    k_dd, x, phi_dd, _ = dd_sn(aniso, 16, cells_per_mfp=80)
    assert r.k == pytest.approx(k_dd, abs=2e-6)
    np.testing.assert_allclose(np.interp(x, r.x, r.phi), phi_dd, rtol=2e-4)


def test_anisotropy_changes_the_answer(aniso, cell):
    """Forward scattering in the moderator matters: ~50 pcm in this cell."""
    dk = sn_even_parity(aniso, 16).k - sn_even_parity(cell, 16).k
    assert -1e-3 < dk < -1e-4


def test_anisotropic_neutron_balance(aniso):
    r = sn_even_parity(aniso, 8)
    t = np.array([z.thickness for z in aniso.zones])
    prod = sum(z.nu_sigma_f * a * w for z, a, w in zip(aniso.zones, r.zone_avg, t))
    absn = sum(z.sigma_a * a * w for z, a, w in zip(aniso.zones, r.zone_avg, t))
    assert r.k == pytest.approx(prod / absn, rel=1e-9)


def test_anisotropy_validation():
    with pytest.raises(ValueError, match="exceeds"):
        Zone(thickness=1, sigma_s=0.5, sigma_c=0.1, sigma_f=0.1, sigma_s_l=(0.6,))
    with pytest.warns(UserWarning, match="negative"):
        Zone(thickness=1, sigma_s=1.2, sigma_c=0.1, sigma_f=0.1, sigma_s_l=(0.8, 0.4, 0.12))
