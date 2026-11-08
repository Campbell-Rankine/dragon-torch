"""RL specialisation of the tuner: a TuneConfig preset and a trial builder for
any DragonRLTrainer. The training-side pieces (StepEvalCallback,
DivergenceGuard) live in trainers.RL.callbacks."""
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict
from trainers.RL import EVAL_STEP_KEY, DragonRLTrainer, DragonRLTrainerConfig, apply_overrides
from .config import TuneConfig
from .objective import TrialSpec


def rl_tune_config(study_name: str, trainer_cls: type[DragonRLTrainer] = DragonRLTrainer,
                   extra_metrics: Optional[dict[str, Any]] = None, **kw) -> TuneConfig:
    """TuneConfig for a DragonRLTrainer study.

    Objective: `trainer_cls.objectives` (the final smoothed evaluation return,
    maximised); pruning compares it at aligned `eval_step`s, so set
    `n_warmup_steps` in env steps (e.g. ~20% of total_timesteps). For a
    multi-objective study pass e.g. `extra_metrics={'steps_per_sec': 'maximize'}`;
    every metric uses the same `final` reduction (default: last value).
    """
    objectives = {**trainer_cls.objectives, **(extra_metrics or {})}
    kw.setdefault('metrics', list(objectives))
    kw.setdefault('directions', list(objectives.values()))
    kw.setdefault('report_key', EVAL_STEP_KEY)
    kw.setdefault('final', 'last')
    return TuneConfig(study_name=study_name, **kw)


class RLTrialBuilder(BaseModel):
    """build(params, trial_dir) for DragonObjective, for any DragonRLTrainer.

    `params` (dotted config paths from a SearchSpace) are applied onto
    `base_config`, checkpoints go to trial_dir/ckpt, and the trainer is returned
    for the objective to fit and close (it also saves the resolved config.json).
    Every trial shares base_config's seeds, so trials differ by hyperparameters
    rather than luck.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trainer_cls: type[DragonRLTrainer]
    base_config: DragonRLTrainerConfig

    def config_for(self, params: dict[str, Any], trial_dir: Path) -> DragonRLTrainerConfig:
        return apply_overrides(self.base_config,
                               {**params, 'model.checkpoint_dir': trial_dir / 'ckpt'})

    def __call__(self, params: dict[str, Any], trial_dir: Path) -> TrialSpec:
        return TrialSpec(trainer=self.trainer_cls(self.config_for(params, trial_dir)))
