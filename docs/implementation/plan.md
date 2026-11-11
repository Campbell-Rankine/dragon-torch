# Replay-buffer algorithms

Reference designs for the off-policy, offline and model-based algorithms planned for dragon-torch. Each entry lists the networks, the policy and the update rule. Everything here trains from a replay buffer.

| | |
|---|---|
| [Shared components](#shared-components) | Heads, target networks and buffer features the algorithms reuse |
| [Discrete control](#discrete-control) | DQN, Double DQN, Dueling, C51, QR-DQN, IQN, Rainbow, Munchausen DQN, R2D2, BBF, SAC-Discrete |
| [Continuous control](#continuous-control) | DDPG, TD3, SAC, TQC, REDQ, DroQ, CrossQ, D4PG, MPO, TD7, BRO, SimBa, MR.Q |
| [Learning from pixels](#learning-from-pixels) | DrQ, DrQ-v2, SPR, CURL |
| [Multi-agent](#multi-agent) | MADDPG, VDN, QMIX, QPLEX, FACMAC |
| [Offline](#offline) | CQL, IQL, TD3+BC, AWAC, ReBRAC, RLPD |
| [Model-based](#model-based) | DreamerV3, TD-MPC2 |
| [Buffer techniques](#buffer-techniques) | PER, HER, n-step returns, resets |

**Notation.** `s`, `a`, `r`, `s'`, `d` form a transition (`d` = terminated). `γ` is the discount factor. `Q̄`, `μ̄`, `V̄` are target networks. `τ` is the Polyak coefficient. `|A|` is the number of discrete actions and `dim(A)` the continuous action dimension.

> **Note**
> Hyperparameters quoted below are the papers' defaults. Treat them as starting points for a [tuning](tuning.md) search space.

## Shared components

### Heads

[`modules/RL/heads.py`](../dragon_torch/modules/RL/heads.py) currently has action-distribution heads only. These algorithms need the following new heads:

| Head | Output | Used by |
|---|---|---|
| `QHead` | `Q(s, ·)`, shape `(B, |A|)` | DQN family, SAC-Discrete, VDN/QMIX agents |
| `DuelingQHead` | `V(s) + A(s, ·) − mean A(s, ·)` | Dueling, Rainbow, R2D2, BBF |
| `CategoricalQHead` | logits `(B, |A|, n_atoms)` on a fixed support | C51, Rainbow, D4PG |
| `QuantileQHead` | `(B, |A|, n_quantiles)` | QR-DQN, TQC, BRO |
| `ImplicitQuantileHead` | `Q(s, ·; τ)` for sampled `τ ~ U(0, 1)` | IQN |
| `StateActionCritic` | `Q(s, a)` from `concat(s, a)` | every continuous actor-critic |
| `DeterministicHead` | `tanh(·)` rescaled to the action bounds | DDPG, TD3, TD7, D4PG, DrQ-v2, MADDPG |
| `SquashedGaussianHead` | `a = tanh(u)`, `u ~ N(μ, σ)`, with the log-prob correction `−Σ log(1 − tanh(u)² + ε)` | SAC and its variants |

### Target networks

Two update rules cover every algorithm:

- **Hard update.** Copy the online weights every `C` gradient steps. DQN uses `C = 10_000` env steps.
- **Polyak update.** `θ̄ ← τθ + (1 − τ)θ̄` after every gradient step. Usually `τ = 0.005`.

### Buffer features

| Feature | Needed by |
|---|---|
| Uniform transition sampling | everything |
| n-step returns | Rainbow, R2D2, BBF, D4PG, DrQ-v2, MR.Q, SPR |
| Priorities (sum-tree) | Rainbow, R2D2, D4PG, TD7 (LAP), MR.Q |
| Sequence sampling with stored recurrent state | R2D2, VDN/QMIX (recurrent agents), DreamerV3, TD-MPC2 |
| Goal relabelling | HER on goal-conditioned envs |
| Fixed dataset plus mixing ratio | offline methods, RLPD |

## Discrete control

All algorithms in this section are off-policy and act greedily on `Q` at evaluation unless the entry says otherwise.

### DQN

Mnih et al., 2013, [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)

**Networks**
- **Q-network**: torso (CNN for pixels, MLP otherwise) → `QHead`.
- **Target**: `Q̄`, hard update.

**Policy**: ε-greedy, with ε annealed linearly (1.0 → 0.01 in the paper's tuned version).

**Update**: `y = r + γ(1 − d) max_a' Q̄(s', a')`, with a Huber loss on `Q(s, a) − y`.

### Double DQN

van Hasselt et al., 2015, [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)

**Networks**: the same as DQN.

**Policy**: ε-greedy.

**Update**: the online network selects the action and the target network evaluates it: `y = r + γ(1 − d) Q̄(s', argmax_a' Q(s', a'))`.

### Dueling DQN

Wang et al., 2015, [arXiv:1511.06581](https://arxiv.org/abs/1511.06581)

**Networks**: a shared torso → `DuelingQHead` with separate `V(s)` and `A(s, a)` streams, combined as `Q = V + A − mean_a A`.

**Policy**: ε-greedy.

**Update**: the Double DQN target. Only the head changes.

### C51

Bellemare et al., 2017, [arXiv:1707.06887](https://arxiv.org/abs/1707.06887)

**Networks**: torso → `CategoricalQHead` with 51 atoms on `[V_min, V_max] = [−10, 10]`, softmaxed per action.

**Policy**: ε-greedy on the expected value `Σ_i z_i p_i(s, a)`.

**Update**: shift the target distribution by `r + γz`, project it back onto the support, and minimise the cross-entropy against the online distribution.

### QR-DQN

Dabney et al., 2017, [arXiv:1710.10044](https://arxiv.org/abs/1710.10044)

**Networks**: torso → `QuantileQHead` with `N = 200` quantiles at the fixed midpoints `τ̂_i = (2i − 1) / 2N`.

**Policy**: ε-greedy on the quantile mean.

**Update**: quantile Huber loss between all pairs of online and target quantiles.

### IQN

Dabney et al., 2018, [arXiv:1806.06923](https://arxiv.org/abs/1806.06923)

**Networks**: torso features `ψ(s)` ⊙ a cosine embedding `φ(τ) = ReLU(Σ_j cos(πjτ) w_j + b)` with 64 cosines → `QHead`. `τ` is sampled for each forward pass.

**Policy**: ε-greedy on the mean over `K = 32` sampled `τ`. A distorted `τ` gives a risk-sensitive policy.

**Update**: quantile Huber loss over `N = N' = 64` sampled `τ`.

### Rainbow

Hessel et al., 2017, [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)

**Networks**: torso → dueling C51 head built from **NoisyLinear** layers (factorised Gaussian noise).

**Policy**: greedy on the noisy network. The noise replaces ε-greedy.

**Update**: C51 projection with Double-DQN action selection and 3-step returns, sampled with proportional PER.

### Munchausen DQN

Vieillard et al., 2020, [arXiv:2007.14430](https://arxiv.org/abs/2007.14430)

**Networks**: the same as DQN.

**Policy**: `π = softmax(Q / τ_ent)` with `τ_ent = 0.03`. Act ε-greedy or sample from `π`.

**Update**: the reward gets the bonus `α · clip(τ_ent log π(a|s), l_0, 0)`, and the target is the soft value `Σ_a' π̄(a'|s')(Q̄(s', a') − τ_ent log π̄(a'|s'))`. Defaults: `α = 0.9`, `l_0 = −1`.

### R2D2

Kapturowski et al., ICLR 2019 ([OpenReview](https://openreview.net/forum?id=r1lyTjAqYX))

**Networks**: CNN → LSTM (512) → `DuelingQHead`, plus a target copy.

**Policy**: ε-greedy, with a different ε for each distributed actor.

**Update**: replays stored sequences of 80 steps, overlapping by 40. The first 40 steps are a burn-in that rebuilds the LSTM state from the stored initial state. Then Double-DQN n-step targets with value rescaling `h(x) = sign(x)(√(|x| + 1) − 1) + εx`. Priorities mix the max and mean TD error over the sequence.

### BBF

Schwarzer et al., 2023, [arXiv:2305.19452](https://arxiv.org/abs/2305.19452)

**Networks**: an Impala-CNN torso scaled 4× in width, a dueling distributional head and the SPR auxiliary modules (see [SPR](#spr)). The target is an EMA of the online network.

**Policy**: ε-greedy.

**Update**: replay ratio 8, with periodic shrink-and-perturb resets of the network. The n-step length is annealed from 10 to 3 and `γ` from 0.97 to 0.997 after each reset. Adds weight decay.

### SAC-Discrete

Christodoulou, 2019, [arXiv:1910.07207](https://arxiv.org/abs/1910.07207)

**Networks**
- **Actor**: torso → `CategoricalHead` giving `π(·|s)`.
- **Critics**: two `QHead`s with Polyak targets.

**Policy**: a stochastic categorical policy, sampled during training.

**Update**: expectations over actions are exact sums, with no reparameterisation. The critic target is `r + γ Σ_a' π(a'|s')(min_j Q̄_j(s', a') − α log π(a'|s'))`. `α` is tuned automatically towards a target entropy of `0.98 · log|A|`.

## Continuous control

All algorithms in this section are off-policy. At evaluation, deterministic actors output `μ(s)` and stochastic actors output `tanh(mean)`.

### DDPG

Lillicrap et al., 2015, [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)

**Networks**
- **Actor**: MLP → `DeterministicHead`, giving `μ(s)`.
- **Critic**: `StateActionCritic`, with the action fed in at the second layer.
- **Targets**: `μ̄` and `Q̄`, with Polyak `τ = 0.001`.

**Policy**: deterministic. The behaviour policy is `μ(s)` plus Ornstein–Uhlenbeck noise (`θ = 0.15`, `σ = 0.2`).

**Update**: critic target `y = r + γ(1 − d) Q̄(s', μ̄(s'))`. The actor maximises `Q(s, μ(s))`.

### TD3

Fujimoto et al., 2018, [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)

**Networks**
- **Actor**: MLP (256, 256) → `DeterministicHead`.
- **Critics**: two `StateActionCritic`s.
- **Targets**: `μ̄`, `Q̄_1` and `Q̄_2`, with Polyak `τ = 0.005`.

**Policy**: deterministic. The behaviour policy is `μ(s) + N(0, 0.1)`.

**Update**
- **Clipped double Q**: `y = r + γ(1 − d) min_j Q̄_j(s', ã)`.
- **Target policy smoothing**: `ã = clip(μ̄(s') + clip(N(0, 0.2), −0.5, 0.5), a_low, a_high)`.
- **Delayed updates**: the actor and targets update once every 2 critic updates. The actor maximises `Q_1(s, μ(s))`.

### SAC

Haarnoja et al., 2018, [arXiv:1801.01290](https://arxiv.org/abs/1801.01290), [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)

**Networks**
- **Actor**: MLP (256, 256) → `SquashedGaussianHead`, with `log σ` clamped to `[−20, 2]`.
- **Critics**: two `StateActionCritic`s with Polyak targets (`τ = 0.005`). There is no target actor.

**Policy**: stochastic maximum-entropy. Training samples `a ~ π(·|s)`.

**Update**
- **Critic**: `y = r + γ(1 − d)(min_j Q̄_j(s', a') − α log π(a'|s'))` with `a' ~ π(·|s')`.
- **Actor**: minimise `E[α log π(ã|s) − min_j Q_j(s, ã)]` through the reparameterised `ã`.
- **Temperature**: minimise `−α(log π + H̄)` with `H̄ = −dim(A)`.

### TQC

Kuznetsov et al., 2020, [arXiv:2005.04269](https://arxiv.org/abs/2005.04269)

**Networks**: a SAC actor, and `N = 5` critics that each output `M = 25` quantiles of `Z(s, a)`.

**Policy**: the same as SAC.

**Update**: pool all `N·M` target atoms, sort them and drop the top `k·N` (`k = 2` per critic), then apply the quantile Huber loss to the rest. Dropping the top atoms controls overestimation more finely than a min over critics. The actor uses the mean of all atoms.

### REDQ

Chen et al., 2021, [arXiv:2101.05982](https://arxiv.org/abs/2101.05982)

**Networks**: a SAC actor and an ensemble of `N = 10` critics.

**Policy**: the same as SAC.

**Update**: update-to-data ratio `G = 20`. The target uses the min over a random subset of `M = 2` target critics. The actor uses the mean over all `N` critics.

### DroQ

Hiraoka et al., 2021, [arXiv:2110.02034](https://arxiv.org/abs/2110.02034)

**Networks**: a SAC actor and two critics, with each hidden layer `Linear → Dropout(0.01) → LayerNorm → ReLU`.

**Policy**: the same as SAC.

**Update**: the SAC update at `UTD = 20`. Dropout replaces REDQ's large ensemble.

### CrossQ

Bhatt et al., ICLR 2024, [arXiv:1902.05605](https://arxiv.org/abs/1902.05605)

**Networks**: a SAC actor and two wide critics (2048 units) with **batch renormalisation**. There are **no target networks**.

**Policy**: the same as SAC.

**Update**: `(s, a)` and `(s', a')` are concatenated into one batch for a joint critic forward pass, so the batch-norm statistics cover both distributions. The target half is detached. It reaches high sample efficiency at `UTD = 1`.

### D4PG

Barth-Maron et al., 2018, [arXiv:1804.08617](https://arxiv.org/abs/1804.08617)

**Networks**: a deterministic actor and a `StateActionCritic` with a categorical (C51-style) output, plus Polyak targets.

**Policy**: deterministic with Gaussian exploration noise, run on many distributed actors.

**Update**: the categorical distributional Bellman projection with N-step returns and PER. The actor gradient uses the critic's expected value.

### MPO

Abdolmaleki et al., 2018, [arXiv:1806.06920](https://arxiv.org/abs/1806.06920)

**Networks**: a Gaussian actor (mean and Cholesky factor), a `StateActionCritic` and a target actor and critic.

**Policy**: stochastic Gaussian.

**Update**
- **Critic**: Retrace targets.
- **E-step**: sample actions from the target policy and weight them by `exp(Q / η)`, with `η` set by a KL bound `ε`.
- **M-step**: fit the parametric policy by weighted maximum likelihood under separate KL constraints on the mean and the covariance.

### TD7

Fujimoto et al., 2023, [arXiv:2306.02451](https://arxiv.org/abs/2306.02451)

**Networks**
- **SALE encoder**: `f(s) → z_s` and `g(z_s, a) → z_sa`, trained so that `z_sa` predicts `z_{s'}`.
- **Actor and twin critics** (TD3-style) that take the embeddings as extra input. Inputs are normalised with AvgL1Norm.

**Policy**: deterministic with Gaussian noise. Evaluation uses a **policy checkpoint**, the best policy seen over recent episodes.

**Update**: TD3 updates with LAP (loss-adjusted prioritised replay) and clipped value targets. Adding a BC term gives offline TD7.

### BRO

Nauman et al., 2024, [arXiv:2405.16158](https://arxiv.org/abs/2405.16158)

**Networks**: a **BroNet** critic (residual blocks with LayerNorm, scaled to about 5M parameters) with quantile outputs, and a BroNet actor.

**Policy**: two actors, an optimistic exploration actor for data collection and a pessimistic actor for evaluation.

**Update**: SAC-style at `UTD` up to 10, with periodic full resets and weight decay.

### SimBa

Lee et al., 2024, [arXiv:2410.09754](https://arxiv.org/abs/2410.09754). SimbaV2: [arXiv:2502.15280](https://arxiv.org/abs/2502.15280)

**Networks**: an architecture change only, applicable to SAC, DDPG or TD-MPC2. Observations pass through running-statistics normalisation, then pre-LayerNorm residual MLP blocks, then a final LayerNorm. SimbaV2 adds hyperspherical weight and feature normalisation and a distributional critic with reward scaling.

**Policy**: inherited from the base algorithm.

**Update**: the base algorithm's update. SimBa scales the network size without needing resets.

### MR.Q

Fujimoto et al., 2025, [arXiv:2501.16142](https://arxiv.org/abs/2501.16142)

**Networks**: an encoder producing `z_s` and `z_sa`, trained on a model-based loss (predicting reward, next embedding and termination). A deterministic actor and twin critics sit on top of the embeddings.

**Policy**: deterministic with Gaussian noise. Designed to use one set of hyperparameters for Gym, DMC and Atari.

**Update**: the encoder trains on short unrolled model rollouts. The critics use multi-step returns and LAP-style prioritised replay. Only the encoder update is model-based.

## Learning from pixels

### DrQ

Kostrikov et al., 2020, [arXiv:2004.13649](https://arxiv.org/abs/2004.13649)

**Networks**: a shared CNN encoder → a SAC actor and critics. Only critic gradients update the encoder.

**Policy**: the same as SAC.

**Update**: random-shift augmentation (pad by 4, then random crop). Q-targets are averaged over `K` augmentations and Q-estimates over `M`.

### DrQ-v2

Yarats et al., 2021, [arXiv:2107.09645](https://arxiv.org/abs/2107.09645)

**Networks**: a CNN encoder → a DDPG actor (deterministic) and twin critics.

**Policy**: deterministic, with exploration noise σ decayed on a schedule (1.0 → 0.1).

**Update**: DDPG with clipped double Q, 3-step returns and random-shift augmentation.

### SPR

Schwarzer et al., 2020, [arXiv:2007.05929](https://arxiv.org/abs/2007.05929)

**Networks**: a Rainbow-style Q-network plus a convolutional transition model and projection heads. The target encoder is an EMA of the online encoder.

**Policy**: inherited from the base Q-learner.

**Update**: the RL loss plus an auxiliary loss. The transition model predicts latents `K = 5` steps ahead, and the loss is the negative cosine similarity to the target encoder's projections of the real future observations.

### CURL

Laskin et al., 2020, [arXiv:2004.04136](https://arxiv.org/abs/2004.04136)

**Networks**: a query encoder (shared with RL) and a momentum key encoder, with bilinear similarity `qᵀWk`.

**Policy**: inherited from the base algorithm (SAC or Rainbow).

**Update**: the RL loss plus an InfoNCE loss between two random crops of the same observation.

## Multi-agent

Each entry builds on the agent groups in [Multi-agent RL](multi_agent.md). "Centralised" means the network sees the global state or joint action during training only.

### MADDPG

Lowe et al., 2017, [arXiv:1706.02275](https://arxiv.org/abs/1706.02275)

**Networks**: a deterministic actor `μ_i(o_i)` per agent, a centralised critic `Q_i(x, a_1, …, a_N)` per agent, and Polyak targets.

**Policy**: decentralised deterministic policies with exploration noise.

**Update**: each critic is a DDPG critic on the joint action, with targets that use every agent's target actor. Each actor maximises `Q_i` with respect to its own action only.

### VDN

Sunehag et al., 2017, [arXiv:1706.05296](https://arxiv.org/abs/1706.05296)

**Networks**: per-agent `Q_i(τ_i, a_i)`, usually recurrent (GRU) over the agent's observation history, with shared parameters.

**Policy**: decentralised ε-greedy.

**Update**: `Q_tot = Σ_i Q_i`, trained with a DQN loss on the team reward.

### QMIX

Rashid et al., 2018, [arXiv:1803.11485](https://arxiv.org/abs/1803.11485)

**Networks**: per-agent recurrent `Q_i`, plus a mixing network whose weights come from **hypernetworks** conditioned on the global state. The weights pass through an absolute value so that `∂Q_tot / ∂Q_i ≥ 0`.

**Policy**: decentralised ε-greedy. Because of the monotonicity, each agent's greedy action is also the joint argmax.

**Update**: a DQN loss on `Q_tot` over sampled episodes, with a target network.

### QPLEX

Wang et al., 2020, [arXiv:2008.01062](https://arxiv.org/abs/2008.01062)

**Networks**: per-agent dueling `Q_i`, and a duplex dueling mixer `Q_tot = V_tot + Σ_i λ_i(s, a) A_i`, where `λ_i > 0` comes from multi-head attention.

**Policy**: decentralised ε-greedy.

**Update**: the same as QMIX, but QPLEX can represent the whole IGM function class, not only monotonic mixtures.

### FACMAC

Peng et al., 2020, [arXiv:2003.06709](https://arxiv.org/abs/2003.06709)

**Networks**: per-agent deterministic actors, per-agent critics `Q_i(τ_i, a_i)`, and a non-monotonic mixer (QMIX-style without the absolute value).

**Policy**: decentralised deterministic policies with exploration noise.

**Update**: the critic is trained like QMIX. The actors use a **centralised policy gradient**: all agents' current actions pass through the mixed `Q_tot` together.

## Offline

The buffer is a fixed dataset. These methods differ in how they stop the policy from exploiting `Q` on actions the dataset doesn't contain.

### CQL

Kumar et al., 2020, [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)

**Networks**: SAC for continuous actions, or QR-DQN for discrete.

**Policy**: inherited from the base algorithm.

**Update**: adds `α(E_s[log Σ_a exp Q(s, a)] − E_{(s,a)~D}[Q(s, a)])` to the critic loss. For continuous actions, the log-sum-exp is estimated by importance-sampling uniform and policy actions.

### IQL

Kostrikov et al., 2021, [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)

**Networks**: `V(s)`, twin `Q(s, a)` with a target, and a Gaussian actor.

**Policy**: stochastic, extracted from the critics after (or alongside) value learning.

**Update**
- **V**: expectile regression (`τ = 0.7`) on `Q̄(s, a)` over dataset actions.
- **Q**: `r + γV(s')`.
- **Actor**: advantage-weighted regression with weights `exp(β(Q − V))`.

IQL never queries `Q` on actions outside the dataset.

### TD3+BC

Fujimoto & Gu, 2021, [arXiv:2106.06860](https://arxiv.org/abs/2106.06860)

**Networks**: the same as TD3. States are normalised with dataset statistics.

**Policy**: deterministic.

**Update**: the actor maximises `λQ(s, μ(s)) − (μ(s) − a)²` with `λ = α / mean|Q|` and `α = 2.5`. The critic update is unchanged from TD3.

### AWAC

Nair et al., 2020, [arXiv:2006.09359](https://arxiv.org/abs/2006.09359)

**Networks**: a Gaussian actor and twin critics with targets.

**Policy**: stochastic. AWAC pre-trains offline and then fine-tunes online with the same objective.

**Update**: the critics use a TD target with `a' ~ π`. The actor uses weighted maximum likelihood on buffer actions with weights `exp(A(s, a) / λ)`.

### ReBRAC

Tarasov et al., 2023, [arXiv:2305.09836](https://arxiv.org/abs/2305.09836)

**Networks**: TD3+BC networks with LayerNorm in the critics, deeper networks and large batches.

**Policy**: deterministic.

**Update**: the BC penalty appears in both the actor loss and the critic target, with separate coefficients for each.

### RLPD

Ball et al., 2023, [arXiv:2302.02948](https://arxiv.org/abs/2302.02948)

**Networks**: a SAC actor and an ensemble of 10 LayerNorm critics. The target uses a random subset of 2.

**Policy**: the same as SAC, trained online.

**Update**: **symmetric sampling**. Each batch is 50% offline data and 50% online replay, trained at a high UTD. There's no pre-training and no BC term.

## Model-based

These use a replay buffer of sequences to train a world model, and then train or plan the policy inside that model.

### DreamerV3

Hafner et al., 2023, [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

**Networks**
- **World model (RSSM)**: a deterministic GRU state and a stochastic latent (32 categoricals × 32 classes), an encoder and decoder, and reward and continue heads. Rewards and values are predicted with **symlog two-hot** outputs.
- **Actor and critic**: MLPs on the model state, with an EMA target critic.

**Policy**: stochastic (categorical or Gaussian), trained entirely on imagined rollouts (horizon 15).

**Update**: the world model trains on replayed sequences with reconstruction and KL losses, using free bits and KL balancing. The actor uses REINFORCE (discrete) or reparameterised gradients (continuous) on λ-returns that are normalised by percentiles.

### TD-MPC2

Hansen et al., 2023, [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)

**Networks**: an encoder to a **SimNorm** latent, latent dynamics, a reward head and an ensemble of 5 Q-heads (discrete regression) with no decoder, plus a policy prior.

**Policy**: at each step, plans with **MPPI** over a short horizon in latent space, seeded with samples from the policy prior. Bootstraps with `Q` at the end of the horizon.

**Update**: joint consistency, reward and TD losses over multi-step replayed sub-trajectories. The policy prior trains SAC-style against the Q ensemble.

## Buffer techniques

These are independent of the algorithm, and the buffer should support all of them.

| Technique | Mechanism | Paper |
|---|---|---|
| **Prioritized replay** | Sample with probability `p_i^α / Σ p^α`, where `p_i = |δ_i| + ε` and `α = 0.6`, stored in a sum-tree. Correct the bias with IS weights `(N · P(i))^−β`, annealing `β` from 0.4 to 1. | Schaul et al., 2015, [arXiv:1511.05952](https://arxiv.org/abs/1511.05952) |
| **Hindsight replay** | For goal-conditioned envs, store each transition again with goals swapped for states reached later in the episode ("future" strategy, `k = 4`). | Andrychowicz et al., 2017, [arXiv:1707.01495](https://arxiv.org/abs/1707.01495) |
| **n-step returns** | `Σ_{k<n} γ^k r_{t+k} + γ^n Q̄(s_{t+n}, ·)`, cut short at episode ends. Can be computed when sampling. | Sutton, 1988 (TD(λ)); used in Rainbow |
| **Resets** | Periodically reinitialise the last layers (or the whole network) and keep the buffer. This makes high replay ratios work. | Nikishin et al., 2022, [arXiv:2205.07802](https://arxiv.org/abs/2205.07802) |

## Notes

- **Suggested order.** Start with a replay buffer that supports n-step and priorities, then implement TD3, SAC, DQN/Double DQN, CrossQ, TQC and QMIX. Most of the other entries reuse those networks and only change the head or the target.
- **Sources.** The arXiv IDs and default hyperparameters were written from memory. Check each one against the paper before an implementation depends on it.
