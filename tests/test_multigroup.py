from pathlib import Path

import numpy as np
import pytest

from slab1g import SlabProblem
from slab1g import solve_all as solve_1g
from slab1g.fem import build
from slab1g.multigroup import MGProblem, MGZone, k_infinity, p1, sn, solve_all, sp3
from slab1g.reference import dd_sn_mg

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def as_mg(p: SlabProblem) -> MGProblem:
    zones = [MGZone(z.thickness, [z.sigma_c], [z.sigma_f],
                    np.array([[[z.sigma_sl(l)]] for l in range(z.scattering_order + 1)]),
                    [z.nu], [1.0], z.name) for z in p.zones]
    return MGProblem(zones, p.elements_per_mfp, p.min_elements_per_zone, p.degree)


@pytest.fixture(scope="module")
def coupled():
    """Two groups with down- and up-scatter, P3 transfers and fission in both groups."""
    S0 = np.array([[0.50, 0.04], [0.002, 1.2]])
    S1 = np.array([[0.20, 0.01], [0.0005, 0.3]])
    S2 = np.array([[0.05, 0.002], [0.0, 0.06]])
    S3 = np.array([[0.01, 0.0], [0.0, 0.01]])
    fuel = MGZone(0.6, [0.01, 0.08], [0.003, 0.15], np.stack([0.6 * S0, 0.3 * S1, 0.2 * S2, 0.2 * S3]),
                  [2.6, 2.43], [0.97, 0.03], "fuel")
    mod = MGZone(1.1, [0.0005, 0.02], [0, 0], np.stack([S0, S1, S2, S3]), [2.43, 2.43], [1, 0], "mod")
    return MGProblem([fuel, mod], 8, 16, 2)


def test_one_group_limit_matches_one_group_code():
    p = SlabProblem.from_toml(EXAMPLES / "anisotropic_cell.toml")
    for a, b in zip(solve_1g(p, (8,)), solve_all(as_mg(p), (8,))):
        assert b.k == pytest.approx(a.k, rel=1e-11)
        np.testing.assert_allclose(b.phi[0], a.phi, rtol=1e-10)


def test_s2_is_p1_and_s4_is_sp3_with_group_coupling(coupled):
    d = build(coupled)
    r1, r3 = p1(coupled, d), sp3(coupled, d)
    with pytest.warns(UserWarning, match="dropped"):
        s2 = sn(coupled, 2, d)
    s4 = sn(coupled, 4, d)
    assert s2.k == pytest.approx(r1.k, rel=1e-10)
    assert s4.k == pytest.approx(r3.k, rel=1e-10)
    np.testing.assert_allclose(s2.phi, r1.phi, rtol=1e-8)
    np.testing.assert_allclose(s4.phi, r3.phi, rtol=1e-8)


def test_infinite_medium(coupled):
    fuel = coupled.zones[0]
    p = MGProblem([fuel, fuel], 8, 16, 2)
    kinf = k_infinity(fuel)
    for r in solve_all(p, (8,)):
        assert r.k == pytest.approx(kinf, rel=1e-10)
        for g in range(2):
            assert np.ptp(r.phi[g]) < 1e-8 * r.phi[g].max()


def test_neutron_balance(coupled):
    """No leakage: k = total production / total absorption, summed over groups."""
    t = np.array([z.thickness for z in coupled.zones])
    for r in solve_all(coupled, (8,)):
        prod = sum((z.nu_sigma_f * r.zone_avg[:, i]).sum() * t[i] for i, z in enumerate(coupled.zones))
        absn = sum((z.sigma_a_g * r.zone_avg[:, i]).sum() * t[i] for i, z in enumerate(coupled.zones))
        assert r.k == pytest.approx(prod / absn, rel=1e-9)


def test_matches_independent_multigroup_dd():
    S0 = np.array([[0.50, 0.04], [0.002, 1.2]])
    S1 = np.array([[0.20, 0.01], [0.0005, 0.3]])
    fuel = MGZone(0.6, [0.01, 0.08], [0.003, 0.15], np.stack([0.6 * S0, 0.3 * S1]), [2.6, 2.43], [0.97, 0.03])
    mod = MGZone(1.1, [0.005, 0.12], [0, 0], np.stack([0.7 * S0, 0.7 * S1]), [2.43, 2.43], [1, 0])
    p = MGProblem([fuel, mod], 8, 16, 2)
    r = sn(p, 8)
    k, x, phi, _ = dd_sn_mg(p, 8, cells_per_mfp=80)
    assert r.k == pytest.approx(k, abs=3e-6)
    fe = np.array([np.interp(x, r.x, r.phi[g]) for g in range(2)])
    np.testing.assert_allclose(fe / fe.sum(), phi / phi.sum(), rtol=1e-4)


def test_fission_normalisation(coupled):
    t = np.array([z.thickness for z in coupled.zones])
    for r in solve_all(coupled, (8,), norm="fission"):
        rate = sum((z.sigma_f * r.zone_avg[:, i]).sum() * t[i] for i, z in enumerate(coupled.zones))
        assert rate == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("name", ["pincell_2g", "reactor_2g", "uox_mox_2g", "uox_rodded_2g"])
def test_two_group_examples_run(name):
    p = MGProblem.from_toml(EXAMPLES / f"{name}.toml")
    assert p.G == 2
    res = solve_all(p, (8,))
    assert all(np.all(r.phi > 0) for r in res)


def test_input_validation():
    with pytest.raises(ValueError):
        MGZone(1.0, [0.1, 0.1], [0, 0], np.array([[[0.5, 0.1], [0, 1.0]], [[0.6, 0], [0, 0.1]]]), [2.4, 2.4], [1, 0])
    with pytest.raises(ValueError):
        MGZone(1.0, [0.1, 0.1, 0.1], [0, 0], np.array([[0.5, 0.1], [0, 1.0]]), [2.4, 2.4], [1, 0])
