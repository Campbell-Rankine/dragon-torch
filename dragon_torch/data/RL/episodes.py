from typing import Any, Iterator, Optional
from dragon_torch.data.envs.config import DragonEnvConfig
from dragon_torch.data.RL.collector import Actor, EnvCollector


class EpisodeSource:
    """Evaluation source: iterating runs the bound model until n_episodes finish,
    yielding {'return', 'shaped_return', 'length'} per episode. 'return' is the
    env's own (raw) reward, before the config's wrappers; 'shaped_return' is
    after them. Neither is affected by reward normalisation.

    Every iteration re-seeds the env with env_cfg.seed, so successive evaluations
    of the same policy are comparable. On bind it borrows the training source's
    observation normaliser (read-only) so the policy sees the inputs it trained on.
    """

    def __init__(self, env_cfg: DragonEnvConfig, n_episodes: int = 10,
                 deterministic: bool = True, max_steps: int = 1_000_000):
        self.collector = EnvCollector(env_cfg, update_stats=False)
        self.n_episodes, self.deterministic, self.max_steps = n_episodes, deterministic, max_steps
        self.actor: Optional[Actor] = None

    def bind(self, model: Actor) -> None:
        if not isinstance(model, Actor):
            raise TypeError(f"EpisodeSource needs a model with act(), got {type(model)}")
        self.actor = model
        train_collector = getattr(getattr(model, 'train_source', None), 'collector', None)
        self.collector.obs_normalizer = getattr(train_collector, 'obs_normalizer', None)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.actor is None:
            raise RuntimeError("EpisodeSource is unbound; evaluate() binds the model")
        self.collector.reset()
        self.collector.drain_episodes()
        yielded, steps = 0, 0
        while yielded < self.n_episodes:
            if steps >= self.max_steps:
                raise RuntimeError(f"{self.n_episodes} episodes did not finish within "
                                   f"{self.max_steps} vector steps; add a TimeLimit wrapper")
            self.collector.step(self.actor, deterministic=self.deterministic)
            steps += 1
            for ep in self.collector.drain_episodes():
                yield {'return': ep.raw_return, 'shaped_return': ep.shaped_return,
                       'length': ep.length}
                yielded += 1
                if yielded == self.n_episodes:
                    return

    @property
    def shaped(self) -> bool:
        """Whether wrappers may make shaped_return differ from return."""
        return self.collector.shaped

    def close(self) -> None:
        self.collector.close()
