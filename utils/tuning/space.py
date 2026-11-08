"""Declarative search spaces. Parameter names are config paths (e.g.
'optim.lr', 'rollout.n_steps') so sampled params apply straight onto a trainer
config with trainers.RL.config.apply_overrides, and read clearly in the dashboard.

    space = SearchSpace(params={
        'optim.lr': FloatParam(low=1e-4, high=3e-3, log=True),
        'rollout.n_steps': CategoricalParam(choices=[64, 128, 256]),
    })
    space = space.fix({'rollout.gamma': 0.99}).narrow({'optim.lr': FloatParam(low=1e-4, high=1e-3, log=True)})
    params = space.suggest(trial)
"""
from typing import Annotated, Any, Literal, Mapping, Union
import optuna
from pydantic import BaseModel, Field, model_validator


class FloatParam(BaseModel):
    kind: Literal['float'] = 'float'
    low: float
    high: float
    log: bool = False
    step: float | None = None

    @model_validator(mode='after')
    def _check(self) -> "FloatParam":
        if self.low > self.high:
            raise ValueError(f"low={self.low} > high={self.high}")
        if self.log and self.low <= 0:
            raise ValueError("log scale needs low > 0")
        return self

    def suggest(self, trial: optuna.Trial, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high, log=self.log, step=self.step)


class IntParam(BaseModel):
    kind: Literal['int'] = 'int'
    low: int
    high: int
    log: bool = False
    step: int = 1

    @model_validator(mode='after')
    def _check(self) -> "IntParam":
        if self.low > self.high:
            raise ValueError(f"low={self.low} > high={self.high}")
        return self

    def suggest(self, trial: optuna.Trial, name: str) -> int:
        return trial.suggest_int(name, self.low, self.high, log=self.log, step=self.step)


class CategoricalParam(BaseModel):
    """Choices must be None / bool / int / float / str (optuna stores them);
    encode structured choices as strings, e.g. hidden sizes as '64x64'."""
    kind: Literal['categorical'] = 'categorical'
    choices: list[Union[None, bool, int, float, str]] = Field(min_length=1)

    def suggest(self, trial: optuna.Trial, name: str) -> Any:
        return trial.suggest_categorical(name, self.choices)


type Param = Annotated[Union[FloatParam, IntParam, CategoricalParam], Field(discriminator='kind')]


class SearchSpace(BaseModel):
    """`params` are sampled each trial; `fixed` values are passed through
    unsampled (they don't appear in trial.params) and win over `params`."""
    params: dict[str, Param] = {}
    fixed: dict[str, Any] = {}

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        out = {name: p.suggest(trial, name) for name, p in self.params.items()
               if name not in self.fixed}
        out.update(self.fixed)
        return out

    def fix(self, values: Mapping[str, Any]) -> "SearchSpace":
        """Copy with these parameters pinned."""
        return self.model_copy(update={'fixed': {**self.fixed, **values}})

    def narrow(self, params: Mapping[str, Param]) -> "SearchSpace":
        """Copy with these parameters' ranges replaced (or added)."""
        return self.model_copy(update={'params': {**self.params, **params}})

    def without(self, *names: str) -> "SearchSpace":
        """Copy without these parameters (the base config's values are used)."""
        return self.model_copy(update={
            'params': {k: v for k, v in self.params.items() if k not in names},
            'fixed': {k: v for k, v in self.fixed.items() if k not in names}})
