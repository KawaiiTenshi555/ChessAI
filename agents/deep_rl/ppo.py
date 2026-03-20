"""
PPO — Proximal Policy Optimization (Schulman et al., 2017)

Re-export from policy_gradient.ppo so that `agents.deep_rl.PPOAgent`
and `agents.policy_gradient.PPOAgent` resolve to the same class.
"""

from ..policy_gradient.ppo import PPOAgent  # noqa: F401

__all__ = ["PPOAgent"]
