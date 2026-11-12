"""The names of version 0.7.1 that later versions renamed still work, each with a DeprecationWarning."""
import warnings

import pytest

import barrierpg
from barrierpg import certificates as C
from barrierpg import convergence as K
from barrierpg import frontier, paper, roles, spec, validation
from barrierpg.examples import plateau, wear

FUNCTIONS = [(C, "price_risk_law", "price_risk_bound"), (C, "risk_floor", "risk_lower_bound"),
             (C, "frozen_multiplier_value", "fixed_multiplier_value"), (C, "learning_form_risk", "learned_law_risk"),
             (C, "near_exhaustion_bound", "small_reserve_bound"), (C, "eventual_risk_structural", "ever_failing_bound"),
             (C, "width_for_target", "width_for_tolerance"), (C, "width_for_direct_target", "width_for_direct_tolerance"),
             (validation, "resolution_floor", "zero_failure_bound"), (spec, "failure_tail", "failure_charge"),
             (barrierpg, "failure_tail", "failure_charge"), (plateau, "plateau_initial_law", "initial_law_on_plateau"),
             (wear, "idle_policy", "standing_still_policy")]


@pytest.mark.parametrize("module,old,new", FUNCTIONS)
def test_old_function_names(module, old, new):
    with pytest.warns(DeprecationWarning, match=new):
        assert getattr(module, old) is getattr(module, new)


def test_new_names_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert C.price_risk_bound(1.0, 0.95, 5.43).value <= 1.0
        assert validation.zero_failure_bound(100, 0.05) > 0
        assert K.multiplier_diagnosis([dict(multiplier=0.0, failure_rate=0.01, episodes=400) for _ in range(60)], 0.05, 0.95).status == "inactive"


def test_unknown_names_still_fail():
    with pytest.raises(AttributeError):
        C.no_such_function


def test_old_arguments_attributes_and_keys():
    with pytest.warns(DeprecationWarning, match="risk_lower_bound_value"):
        out = C.prescribed_level_attainable(0.05, risk_floor_value=0.01)
    assert out["attainable"] and out["risk_lower_bound"] == 0.01
    with pytest.warns(DeprecationWarning, match="risk_lower_bound"):
        assert out["floor"] == 0.01
    with pytest.warns(DeprecationWarning, match="kappa_drift"):
        assert roles.noise_and_anchor_spread(0.1, kappa_pull=10.0) == roles.noise_and_anchor_spread(0.1, kappa_drift=10.0)
    d = validation.validate_direct(0, 100, 0.05, 0.05)
    with pytest.warns(DeprecationWarning, match="zero_failure_bound"):
        assert d.floor == d.zero_failure_bound
    chk = validation.value_resolution_check(0.95, 100, 10, 10_000, 0.05, 0.05)
    with pytest.warns(DeprecationWarning, match="left_side"):
        assert chk["floor"] == chk["left_side"]
    with pytest.warns(DeprecationWarning, match="truncation_error"):
        assert chk["tail_term"] == chk["truncation_error"]
    mix = roles.RegularizerMix(kappa1=0.1, kappa2=1.0)
    with pytest.warns(DeprecationWarning, match="drift_strength"):
        assert mix.pull_strength == mix.drift_strength


def test_old_status_and_result_keys():
    status = K.Status("off-tolerance")
    assert status == "off-tolerance" and status != "stable" and hash(status) == hash("off-tolerance")
    with pytest.warns(DeprecationWarning, match="off-tolerance"):
        assert status == "off-target"
    for old, new in paper.RENAMED_KEYS.items():
        with pytest.warns(DeprecationWarning, match=new):
            assert paper.cite(old) == paper.cite(new)
    assert frontier.FrontierCheck([], ["x"], [], [], []).bound_violations == ["x"]
    with pytest.warns(DeprecationWarning, match="bound_violations"):
        assert frontier.FrontierCheck([], ["x"], [], [], []).law_violations == ["x"]
