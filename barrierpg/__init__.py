"""barrierpg: clipped barrier policy gradients under resource budgets, for practitioners.

Companion toolbox to "The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex
Regularization" (Malo, Viitasaari, Suominen, Vilkkumaa, Tahvonen). Keep the learner you use, add a
reward wrapper, and know what each part does: the price sets the risk attained, the interior barrier
buys a margin, the action term holds the policy near a trusted action reference, and the parameter term
makes the learning problem well posed and holds it near a prior policy.

Three questions, three layers:
  configure   BudgetSpec, Requirement, Stage, plan, advise_regularizers, BudgetWrapper,
              callbacks (DualAscent, ActionAnchor, ParameterAnchor, ParameterNoise, BudgetMonitor);
  evidence    evaluate, validation, certificates, frontier, convergence, roles;
  interpret   interpret, regularizer_effects.
The numpy/scipy core needs no learning library; the wrapper uses gymnasium when it is installed and the
callbacks need Stable-Baselines3. Every bound cites its result through barrierpg.paper.cite.
"""
from . import certificates, convergence, frontier, paper, planning, regularizers, report, roles, validation
from .barrier import LogBarrier
from .bounds import Bound
from .evaluate import Evaluation, evaluate, evaluation_from_episodes, sb3_policy, stopped_action_relative_entropy
from .paper import PAPER, cite
from .planning import Plan, plan, when_to_use_another_method
from .regularizers import RegularizerAdvice, RegularizerEffects, advise_regularizers, regularizer_effects
from .report import Report, interpret
from .roles import RegularizerMix
from .spec import (BudgetSpec, Requirement, Stage, discounted_total, failure_tail, multiplier_from_price,
                   price_from_multiplier, price_from_multiplier_physical)
from .wrapper import BudgetWrapper, RunningMaxCost

__version__ = "0.7.0"
__all__ = ["Bound", "BudgetSpec", "Requirement", "Stage", "BudgetWrapper", "RunningMaxCost", "LogBarrier", "Evaluation", "evaluate",
           "evaluation_from_episodes", "sb3_policy", "stopped_action_relative_entropy", "Plan", "plan", "when_to_use_another_method",
           "Report", "interpret", "RegularizerAdvice", "RegularizerEffects", "advise_regularizers", "regularizer_effects",
           "RegularizerMix", "PAPER", "cite", "discounted_total", "failure_tail", "multiplier_from_price", "price_from_multiplier",
           "price_from_multiplier_physical", "certificates", "convergence", "frontier", "paper", "planning", "regularizers", "report",
           "roles", "validation"]
