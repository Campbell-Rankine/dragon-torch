"""Tune PPO on a gymnasium env. Start this same command in several processes (or
machines, with a shared --storage) to run workers in parallel; each joins the study.

    uv run --extra RL --extra torch-cpu python scripts/tune_ppo.py \
        --env CartPole-v1 --study ppo-cartpole --storage journal:runs/optuna.log \
        --timesteps 100000 --eval-every 10000 --n-trials 20

    uv run --extra tune python -m utils.tuning.monitor dashboard --storage journal:runs/optuna.log
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, when run as a script

import torch
from data.envs import DragonEnvConfig
from utils.tuning import run_study
from utils.tuning.presets import PPOBuilder, PPOSpace
from utils.tuning.RL import rl_tune_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default='CartPole-v1')
    p.add_argument('--n-envs', type=int, default=8)
    p.add_argument('--seed', type=int, default=0, help='env / init seed, shared by all trials')
    p.add_argument('--study', default=None, help='default: ppo-<env>')
    p.add_argument('--storage', default='journal:runs/optuna.log')
    p.add_argument('--run-dir', default='runs')
    p.add_argument('--timesteps', type=int, default=100_000)
    p.add_argument('--eval-every', type=int, default=10_000)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--smooth', type=int, default=3, help='evaluations averaged into the objective')
    p.add_argument('--n-trials', type=int, default=20, help='for this worker')
    p.add_argument('--timeout', type=float, default=None, help='seconds, for this worker')
    p.add_argument('--warmup', type=float, default=0.2,
                   help='fraction of --timesteps before pruning starts')
    p.add_argument('--pruner', choices=['median', 'hyperband', 'none'], default='median')
    p.add_argument('--sampler-seed', type=int, default=None,
                   help='default: unseeded, so parallel workers explore different trials')
    p.add_argument('--multi-objective', action='store_true',
                   help='also maximise steps_per_sec (no pruning in multi-objective studies)')
    args = p.parse_args()

    torch.set_num_threads(1)  # many workers x vector envs oversubscribe the CPU otherwise
    cfg = rl_tune_config(
        study_name=args.study or f"ppo-{args.env}", storage=args.storage, run_dir=args.run_dir,
        n_trials=args.n_trials, timeout=args.timeout, pruner=args.pruner, seed=args.sampler_seed,
        n_warmup_steps=int(args.warmup * args.timesteps),
        extra_metrics={'steps_per_sec': 'maximize'} if args.multi_objective else None)
    space = PPOSpace()
    if args.eval_every < max(space.n_steps) * args.n_envs:
        print(f"warning: --eval-every {args.eval_every} < largest epoch "
              f"({max(space.n_steps)} x {args.n_envs} steps); some evaluations will be skipped")
    build = PPOBuilder(env=DragonEnvConfig(env_id=args.env, n_envs=args.n_envs, seed=args.seed),
                       total_timesteps=args.timesteps, eval_every_steps=args.eval_every,
                       eval_episodes=args.eval_episodes, smooth_window=args.smooth)
    print(f"worker pid={os.getpid()} study={cfg.study_name} storage={cfg.storage}")
    study = run_study(cfg, space.suggest, build)
    print(f"done; best: {cfg.study_dir / 'best.json'} ({len(study.trials)} trials in study)")


if __name__ == '__main__':
    main()
