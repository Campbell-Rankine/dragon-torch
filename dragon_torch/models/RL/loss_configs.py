"""Loss settings of each algorithm, as config mixins. An algorithm's model
configs (single- and multi-agent) inherit them, so the settings are model
fields (`model.clip_coef`) whatever the kind of run, and `build_loss()`
makes the single-agent loss."""
from pydantic import BaseModel, Field
from dragon_torch.loss.RL import A2CLoss, PPOLoss


class PPOLossConfig(BaseModel):
    """PPOLoss settings (clipped surrogate, optional value clipping). PPOConfig and
    MultiAgentPPOConfig carry them as model fields."""
    clip_coef: float = Field(default=0.2, gt=0)
    vf_coef: float = Field(default=0.5, ge=0)
    ent_coef: float = Field(default=0.0, ge=0)
    clip_vloss: bool = True
    normalize_advantages: bool = True

    def build_loss(self) -> PPOLoss:
        return PPOLoss(**self.model_dump(include=set(PPOLossConfig.model_fields)))


class A2CLossConfig(BaseModel):
    """A2CLoss settings. A2CConfig and MultiAgentA2CConfig carry
    them as model fields."""
    vf_coef: float = Field(default=0.5, ge=0)
    ent_coef: float = Field(default=0.0, ge=0)
    normalize_advantages: bool = False

    def build_loss(self) -> A2CLoss:
        return A2CLoss(**self.model_dump(include=set(A2CLossConfig.model_fields)))

