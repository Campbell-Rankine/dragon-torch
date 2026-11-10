"""Tune PPO or A2C on a gymnasium env. Start this same command in several processes
(or machines, with a shared --storage) to run workers in parallel; each joins the study.

    uv run --extra RL --extra torch-cpu python scripts/tune_rl.py --algo ppo \
        --env CartPole-v1 --storage journal:runs/optuna.log \
        --timesteps 100000 --eval-every 10000 --n-trials 20

    uv run --extra tune python -m dragon_torch.utils.tuning.monitor dashboard --storage journal:runs/optuna.log

Each trial dir (runs/<study>/trial_NNNN) holds history.jsonl, config.json and
checkpoints; rebuild the best run with
    PPOTrainer(PPOTrainerConfig.model_validate_json(open('.../config.json').read()))
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, when run as a script

import torch
from dragon_torch.data.envs import DragonEnvConfig
from dragon_torch.models.RL import DragonRLConfig
from dragon_torch.trainers.RL import (A2CTrainer, A2CTrainerConfig, EvalConfig, PPOTrainer, PPOTrainerConfig)
from dragon_torch.utils.tuning import run_study
from dragon_torch.utils.tuning.presets import space_for
from dragon_torch.utils.tuning.RL import RLTrialBuilder, rl_tune_config

# each trainer trains one model class (PPOModel, A2CModel); its search-space
# preset is looked up from that class
ALGOS = {
    'ppo': (PPOTrainer, PPOTrainerConfig),
    'a2c': (A2CTrainer, A2CTrainerConfig),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--algo', choices=sorted(ALGOS), default='ppo')
    p.add_argument('--env', default='CartPole-v1')
    p.add_argument('--n-envs', type=int, default=8)
    p.add_argument('--seed', type=int, default=0, help='env / init seed, shared by all trials')
    p.add_argument('--study', default=None, help='default: <algo>-<env>')
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
    trainer_cls, config_cls = ALGOS[args.algo]
    study_name = args.study or f"{args.algo}-{args.env}"
    base = config_cls(
        env=DragonEnvConfig(env_id=args.env, n_envs=args.n_envs, seed=args.seed),
        model=DragonRLConfig(checkpoint_dir=Path(args.run_dir) / study_name, save_every=1_000_000,
                             total_timesteps=args.timesteps),
        eval=EvalConfig(every_steps=args.eval_every, n_episodes=args.eval_episodes,
                        smooth_window=args.smooth),
        seed=args.seed)
    space = space_for(trainer_cls.model_cls)
    largest = max(space.params['rollout.n_steps'].choices) * args.n_envs
    if args.eval_every < largest:
        print(f"warning: --eval-every {args.eval_every} < largest epoch ({largest} steps); "
              "some trials will skip scheduled evaluations")

    cfg = rl_tune_config(
        study_name, trainer_cls, storage=args.storage, run_dir=args.run_dir,
        n_trials=args.n_trials, timeout=args.timeout, pruner=args.pruner, seed=args.sampler_seed,
        n_warmup_steps=int(args.warmup * args.timesteps),
        extra_metrics={'steps_per_sec': 'maximize'} if args.multi_objective else None)
    print(f"worker pid={os.getpid()} study={cfg.study_name} storage={cfg.storage}")
    study = run_study(cfg, space.suggest, RLTrialBuilder(trainer_cls=trainer_cls, base_config=base))
    print(f"done; best: {cfg.study_dir / 'best.json'} ({len(study.trials)} trials in study)")


if __name__ == '__main__':
    main()
