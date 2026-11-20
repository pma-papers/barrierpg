"""Stable-Baselines3 policies for use with the toolbox's budget wrapper."""
from __future__ import annotations

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn

from .networks import ResidualMLP


class ResidualActorCriticPolicy(ActorCriticPolicy):
    """SB3 actor and critic with independent pre-LayerNorm residual MLPs.

    With the default parameter-free flattening, the whole actor backbone belongs to
    the parameter group used by the toolbox's actor regularizers.
    """

    def __init__(self, observation_space, action_space, lr_schedule, *, width: int = 64, depth: int = 2,
                 hidden_width: int | None = None, residual: bool = True, layer_norm: bool = True, **kwargs):
        if kwargs.get("net_arch"):
            raise ValueError("Set width, depth, and hidden_width for ResidualActorCriticPolicy")
        kwargs["net_arch"] = []
        kwargs.setdefault("activation_fn", nn.SiLU)
        self.backbone_kwargs = dict(width=width, depth=depth, hidden_width=hidden_width,
                                    residual=residual, layer_norm=layer_norm)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        extractor = MlpExtractor(self.features_dim, net_arch=[], activation_fn=self.activation_fn, device=self.device)
        extractor.policy_net = ResidualMLP(self.features_dim, activation_fn=self.activation_fn, **self.backbone_kwargs)
        extractor.value_net = ResidualMLP(self.features_dim, activation_fn=self.activation_fn, **self.backbone_kwargs)
        extractor.latent_dim_pi = extractor.latent_dim_vf = self.backbone_kwargs["width"]
        self.mlp_extractor = extractor.to(self.device)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(self.backbone_kwargs, share_features_extractor=self.share_features_extractor)
        return data
