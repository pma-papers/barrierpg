"""Formulate, solve and validate reinforcement-learning policies under resource budgets.

Use a fixed failure price or a reserve penalty with a compatible learner, apply a
constrained actor update, or solve a known finite model by backward induction.
The toolbox also provides finite-scenario policy selection and first-crossing
risk certificates.
"""
from . import certificates, convergence, frontier, paper, planning, regularizers, report, roles, validation
from .alternatives import cpo_step, cpo_update, omnisafe_cpo, select_scenario_policy, upper_cvar
from .barrier import InverseBarrier, LogBarrier, ReserveHinge
from .bounds import Bound
from .convex import categorical_action_divergence, parameter_law_divergence, parameter_potential, regularized_utility
from .dynamic_programming import BudgetDPSolution, Transition, solve_budget_dp
from .evaluate import Evaluation, evaluate, evaluation_from_episodes, sb3_policy, stopped_action_relative_entropy
from .paper import PAPER, cite
from .planning import Plan, plan, when_to_use_another_method
from .regularizers import RegularizerAdvice, RegularizerEffects, advise_regularizers, regularizer_effects
from .report import Report, interpret
from .roles import RegularizerMix
from .spec import (BudgetSpec, Requirement, Stage, discounted_total, failure_charge, multiplier_from_price,
                   price_from_multiplier, price_from_multiplier_physical)
from .wrapper import BudgetWrapper, RunningMaxCost

__version__ = "0.9.0"
__all__ = ["Bound", "BudgetSpec", "Requirement", "Stage", "BudgetWrapper", "RunningMaxCost", "LogBarrier", "Evaluation", "evaluate",
           "evaluation_from_episodes", "sb3_policy", "stopped_action_relative_entropy", "Plan", "plan", "when_to_use_another_method",
           "Report", "interpret", "RegularizerAdvice", "RegularizerEffects", "advise_regularizers", "regularizer_effects",
           "RegularizerMix", "PAPER", "cite", "discounted_total", "failure_charge", "multiplier_from_price", "price_from_multiplier",
           "price_from_multiplier_physical", "certificates", "convergence", "frontier", "paper", "planning", "regularizers", "report",
           "roles", "validation", "InverseBarrier", "ReserveHinge", "cpo_step", "cpo_update", "omnisafe_cpo",
           "select_scenario_policy", "upper_cvar", "categorical_action_divergence", "parameter_potential",
           "parameter_law_divergence", "regularized_utility", "BudgetDPSolution", "Transition", "solve_budget_dp"]


# names of version 0.7.1 and earlier, kept with a DeprecationWarning
from ._deprecation import module_getattr as _module_getattr  # noqa: E402

__getattr__ = _module_getattr(__name__, {"failure_tail": "failure_charge"}, globals())
