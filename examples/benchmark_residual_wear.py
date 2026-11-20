"""Paired PPO pilot that isolates skip connections in a LayerNorm MLP.

From the repository root, run:
    python examples/benchmark_residual_wear.py --output /tmp/residual-wear.json
"""
import argparse
import json
from pathlib import Path

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from barrierpg import Stage, evaluate, sb3_policy
from barrierpg.callbacks import actor_parameters
from barrierpg.examples.wear import WearModel, make_wear_env
from barrierpg.sb3_policies import ResidualActorCriticPolicy


WEAR = WearModel()
STAGE = Stage(m=0.3)
CHECKPOINTS = (8192, 16384, 32768)
SEEDS = (0, 1, 2, 3)
EVAL_EPISODES = 600
EVAL_SEED = 200_000
N_ENVS = 8
PPO_KWARGS = dict(n_steps=128, batch_size=128, n_epochs=4, learning_rate=3e-4, gamma=1.0, device="cpu")
BACKBONE_KWARGS = dict(width=32, depth=2, layer_norm=True, log_std_init=-1.0)


def make_env():
    return make_wear_env(STAGE, WEAR, geometric_stop=True)


def run(seed, residual):
    env = DummyVecEnv([make_env] * N_ENVS)
    try:
        model = PPO(ResidualActorCriticPolicy, env, seed=seed, **PPO_KWARGS,
                    policy_kwargs=dict(BACKBONE_KWARGS, residual=residual))
        # Equal initial action laws and value predictions, irrespective of the backbone.
        for head in (model.policy.action_net, model.policy.value_net):
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
        actor_count = sum(parameter.numel() for parameter in actor_parameters(model))
        total_count = sum(parameter.numel() for parameter in model.policy.parameters())
        result = []
        candidate = sb3_policy(model)
        for index, checkpoint in enumerate(CHECKPOINTS):
            steps = checkpoint - model.num_timesteps
            model.learn(steps, reset_num_timesteps=False)
            eval_seed = EVAL_SEED + EVAL_EPISODES * (len(CHECKPOINTS) * seed + index)
            # Use the same evaluation seeds within a pair without advancing the training RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(eval_seed)
                evidence = evaluate(make_env, candidate, EVAL_EPISODES, spec=WEAR.spec(),
                                    stage=STAGE, seed=eval_seed)
            row = dict(
                seed=seed, residual=residual, steps=model.num_timesteps, eval_seed=eval_seed,
                actor_parameters=actor_count, critic_parameters=total_count - actor_count,
                utility=evidence.U, utility_se=evidence.U_se, failures=evidence.failures,
                risk=evidence.q_beta, value=evidence.V_m, value_se=evidence.V_m_se,
            )
            print(json.dumps(row), flush=True)
            result.append(row)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [row for seed in SEEDS for residual in (False, True) for row in run(seed, residual)]
    if args.output:
        record = dict(
            software=dict(torch=torch.__version__, stable_baselines3=stable_baselines3.__version__,
                          gymnasium=gymnasium.__version__, numpy=np.__version__),
            failure_price=STAGE.m, eval_episodes=EVAL_EPISODES, n_envs=N_ENVS,
            ppo=PPO_KWARGS, backbone=BACKBONE_KWARGS, results=rows,
        )
        args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
