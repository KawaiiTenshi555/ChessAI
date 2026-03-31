# Agent Review Changelog

Date: 2026-03-20

This document summarizes the deep review pass done per AI agent implementation, the concrete code changes applied in this session, and the main follow-up improvements that still matter.

## 2026-03-26 Hotfix

Fixed a PPO training crash triggered after repeated hyperparameter edits in the web UI:

- Root cause: once a PPO integer field like `ppo_epochs`, `batch_size`, or `rollout_steps` had been polluted into a `float` in memory, later updates could keep using that wrong runtime type and training would crash on `range(self.ppo_epochs)`.
- Fix: `web/app.py` now normalizes agent runtime config from the agent class `DEFAULT_CONFIG` schema instead of trusting the current in-memory types.
- Fix: PPO and other agents are normalized when loaded from the registry, after checkpoint load, before training, and after hyperparameter updates.
- Fix: added regression tests to ensure polluted PPO integer fields are converted back to real integers before use.
- Fix: `agents/policy_gradient/ppo.py` now handles single-step or very small final rollouts safely by keeping value tensors 1D and using `advantages.std(unbiased=False)` during normalization.
- Fix: `web/app.py` now marks training as running before the worker thread starts and protects that transition with a lock, which prevents multiple rapid training launches from starting concurrent PPO runs on the same agent.

## Scope

Reviewed code ownership across:

- `agents/tabular/sarsa.py`
- `agents/tabular/q_learning.py`
- `agents/tabular/expected_sarsa.py`
- `agents/tabular/monte_carlo.py`
- `agents/deep_rl/dqn.py`
- `agents/policy_gradient/reinforce.py`
- `agents/policy_gradient/ppo.py`
- `agents/deep_rl/ppo.py`
- `agents/base_agent.py`
- `web/app.py`
- `web/templates/index.html`
- `tests/test_agents.py`

## Applied In This Session

- Added automatic checkpoint directory creation in `BaseAgent.save()`.
- Added `BaseAgent.finalize_training()` and called it at the end of the shared training loop.
- Added `PPOAgent.finalize_training()` so partial rollouts are flushed instead of being dropped when training ends.
- Reworked `web/app.py` agent creation to lazy-instantiation instead of building every agent up front.
- Added automatic checkpoint loading from `models/<agent>.pkl` when an agent is first created in the web app.
- Added automatic checkpoint saving after web training completes.
- Added automatic checkpoint saving after hyperparameter updates succeed.
- Rebuilt web agent creation from saved config before loading checkpoint state. This prevents shape mismatches for agents like DQN/PPO when a saved config changes architecture-related settings.
- Made `/api/hyperparams` type-aware instead of coercing almost everything to `float`.
- Applied live `lr` updates to optimizers when supported.
- Rejected structural hot-edits such as `hidden_sizes` and DQN `buffer_size` from the web API instead of silently corrupting runtime state.
- Blocked `/api/select_agent` while background training is running to reduce status drift and state corruption.
- Recorded real episode reward/length inside the web training loop instead of always passing `0.0, 0` to `on_episode_end()`.
- Added training error capture and checkpoint status fields to the web training status payload.
- Updated the web UI to show when an agent was loaded from checkpoint and when training finished with checkpoint save.

## Validation

Validated with:

- `python -m py_compile agents\base_agent.py agents\policy_gradient\ppo.py web\app.py tests\test_web_persistence.py`
- `python -m pytest tests\test_agents.py tests\test_web_persistence.py -q`

Result:

- `47 passed`

## New Tests Added

Added `tests/test_web_persistence.py` covering:

- Q-Learning checkpoint reload from the web app helper path
- DQN recreation from saved config before checkpoint load
- Type-safe DQN hyperparameter persistence through the web API
- Rejection of structural DQN hot-edits
- PPO partial-rollout flush at end of training

## Agent Review Summary

### SARSA

Current logic:

- Clean on-policy tabular SARSA with `_next_action` carry-over between `update()` and `select_action()`.
- Persistence works for `Q` and `epsilon`.

Main remaining improvements:

- Raw `obs.tobytes()` keys make the state space too large for serious chess performance.
- Greedy tie-breaking depends on move ordering.
- Live play and training still share the same mutable instance in the web app.
- Black-side training/inference viewpoint consistency still needs a clear product contract.

### Q-Learning

Current logic:

- Standard off-policy TD update over legal actions only.
- Web lifecycle is improved by lazy loading and automatic checkpoints.

Main remaining improvements:

- Checkpoint load still does not generically restore full config through `BaseAgent.load()` for non-web callers.
- Black-side web inference may still differ from the training convention.
- Empty-legal-action defensive checks are still worth adding inside the agent.
- The shared live/training instance still lacks real locking.

### Expected SARSA

Current logic:

- Correct epsilon-greedy expectation update and existing algorithm tests.
- Web persistence and lazy loading now improve usability.

Main remaining improvements:

- Full config restore is still not generic outside the web creation path.
- Promotion decoding parity between web play and `ChessEnv` should be unified.
- Hyperparameter range validation is still worth enforcing more strictly.
- Shared agent concurrency remains open.

### Monte Carlo

Current logic:

- Episode buffering and flush behavior are correct and tested.
- Web lifecycle now checkpoints automatically like the other agents.

Main remaining improvements:

- Full checkpoint fidelity for config is still incomplete outside config-aware web recreation.
- Web promotion decoding should match the environment exactly.
- Black-side observation consistency remains a design issue.
- Shared agent concurrency remains open.

### DQN

Current logic:

- Replay-buffer DQN with legal-action masking, target network, and persistence.
- Web hot-update path is now much safer: ints stay ints, `lr` updates reach the optimizer, and structural edits are rejected.

Main remaining improvements:

- Replay buffer is not persisted, so training resume is only partial.
- The same DQN instance is still shared between training and gameplay without a lock.
- Live play still uses epsilon-greedy action selection instead of a pure evaluation policy.
- The UI still exposes replay-buffer size through the legacy `q_table_size` field name.

### REINFORCE

Current logic:

- Policy-gradient implementation is structurally sound and already persisted at the model level.
- Shared web hot-update handling is now better for `lr` and safer for unsupported edits.

Main remaining improvements:

- Full config restoration still is not generic outside config-aware web recreation.
- Shared gameplay/training concurrency is still open.
- REINFORCE-specific regression coverage is still thinner than the tabular suite.
- Live play still uses stochastic sampling instead of a deterministic evaluation mode.

### PPO

Current logic:

- Actor-critic PPO with GAE, masked action selection, and rollout updates.
- Partial rollout loss at the end of training is fixed in this session.
- Web checkpoint loading now recreates PPO from saved config before loading weights.

Main remaining improvements:

- Shared gameplay/training concurrency is still open.
- Mid-rollout checkpoint resume is not supported.
- The actor head still scores all `4096` actions even though only a small subset is legal.
- PPO-specific test coverage is still smaller than ideal even after adding end-of-training flush coverage.

## Recommended Next Steps

1. Add a dedicated lock or separate eval/training agent instances in `web/app.py`.
2. Unify promotion decoding and black-side observation/action conventions between the web board path and `ChessEnv`.
3. Extend persistence so full saved config is restored consistently for all non-web load paths too.
4. Add direct regression suites for DQN, REINFORCE, and PPO beyond the new persistence tests.
5. Decide whether live AI play should use exploration/stochastic sampling or a deterministic best-move evaluation policy.
