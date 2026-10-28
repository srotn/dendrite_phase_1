"""Shared learnable discrete reciprocal selectors for Dendrite experiments."""

from __future__ import annotations

import math

import torch
from torch import nn


SAFE_RECIPROCAL_EPSILON = 0.10
SAFE_RECIPROCAL_CLIP = 6.0


def stable_reciprocal(x: torch.Tensor) -> torch.Tensor:
    """Finite reciprocal shared by every selectable input.

    Values near zero are mapped to zero and all other reciprocals are clipped;
    this makes the discrete model finite on every historical task without
    depending on a model-specific data filter.
    """
    magnitude = x.abs().clamp_min(SAFE_RECIPROCAL_EPSILON)
    reciprocal = x.sign() / magnitude
    reciprocal = torch.where(x.abs() < SAFE_RECIPROCAL_EPSILON, torch.zeros_like(reciprocal), reciprocal)
    return reciprocal.clamp(-SAFE_RECIPROCAL_CLIP, SAFE_RECIPROCAL_CLIP)


class _DDStack(nn.Module):
    def __init__(self, width: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = x
        h = x
        for layer in self.layers:
            h = layer(h) * anchor
        return h


class LearnableDiscreteDD(nn.Module):
    """DD with one trainable binary raw/reciprocal choice per input.

    ``selector_logits`` start positive, so the first forward pass is exactly
    the raw input. The hard sign is used in the forward path; a tanh surrogate
    supplies the straight-through gradient. During early epochs a decaying
    epsilon exploration mask samples hard signs, allowing both discrete values
    to be visited while the logits and DD weights train jointly.
    """

    def __init__(self, input_width: int, layers: int, output_dim: int, name: str, *, seed: int = 0):
        super().__init__()
        self.name = name
        # logit=+1 gives a hard +1 choice at initialization while retaining a
        # useful STE derivative; a saturated value such as +4 would prevent
        # the selector from exploring the -1 branch.
        self.selector_logits = nn.Parameter(torch.ones(input_width))
        self.dd = _DDStack(input_width, layers)
        self.output = nn.Linear(input_width, output_dim)
        self._epoch = 0
        self._total_epochs = 500
        self._seed = int(seed)

    def set_epoch(self, epoch: int, total_epochs: int = 500) -> None:
        self._epoch = int(epoch)
        self._total_epochs = max(int(total_epochs), 1)

    def _hard_selection(self) -> torch.Tensor:
        hard = torch.where(self.selector_logits >= 0, torch.ones_like(self.selector_logits), -torch.ones_like(self.selector_logits))
        if not self.training:
            return hard
        progress = min(max((self._epoch - 1) / max(self._total_epochs - 1, 1), 0.0), 1.0)
        epsilon = 0.20 * (1.0 - progress) + 0.02 * progress
        if epsilon <= 0:
            return hard
        generator = torch.Generator(device=self.selector_logits.device)
        generator.manual_seed(self._seed + self._epoch * 1009)
        explore = torch.rand(self.selector_logits.shape, generator=generator, device=self.selector_logits.device) < epsilon
        random_sign = torch.where(
            torch.rand(self.selector_logits.shape, generator=generator, device=self.selector_logits.device) < 0.5,
            -torch.ones_like(hard),
            torch.ones_like(hard),
        )
        return torch.where(explore, random_sign, hard)

    def selection_values(self) -> torch.Tensor:
        return torch.where(self.selector_logits.detach() >= 0, torch.ones_like(self.selector_logits), -torch.ones_like(self.selector_logits))

    def selection_probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.selector_logits.detach())

    @torch.no_grad()
    def initialize_selection(self, *, random_start: bool, seed: int = 0) -> None:
        """Initialize hard choices, preserving the explicit +1 default."""
        if not random_start:
            self.selector_logits.fill_(1.0)
            return
        generator = torch.Generator(device=self.selector_logits.device).manual_seed(int(seed))
        signs = torch.where(
            torch.rand(self.selector_logits.shape, generator=generator, device=self.selector_logits.device) < 0.5,
            -torch.ones_like(self.selector_logits),
            torch.ones_like(self.selector_logits),
        )
        self.selector_logits.copy_(signs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hard = self._hard_selection()
        temperature = 1.5 - 0.9 * min(max((self._epoch - 1) / max(self._total_epochs - 1, 1), 0.0), 1.0)
        soft = torch.tanh(self.selector_logits / temperature)
        selection = hard.detach() - soft.detach() + soft
        # Arithmetic mixing preserves the hard +/-1 forward choice while
        # retaining the tanh straight-through derivative in backward.
        positive = (selection.unsqueeze(0) + 1.0) * 0.5
        transformed = positive * x + (1.0 - positive) * stable_reciprocal(x)
        return self.output(self.dd(transformed))


class GumbelDiscreteDD(nn.Module):
    """Hard Bernoulli/Gumbel selector with annealed entropy exploration."""

    def __init__(self, input_width: int, layers: int, output_dim: int, name: str, *, seed: int, total_epochs: int = 500):
        super().__init__()
        self.name = name
        self.selector_logits = nn.Parameter(torch.randn(input_width) * 0.05)
        self.dd = _DDStack(input_width, layers)
        self.output = nn.Linear(input_width, output_dim)
        self.epoch = 1
        self.total_epochs = total_epochs
        self.seed = int(seed)

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.epoch = int(epoch)
        if total_epochs is not None:
            self.total_epochs = int(total_epochs)

    def temperature(self) -> float:
        progress = min(max((self.epoch - 1) / max(self.total_epochs - 1, 1), 0.0), 1.0)
        return 2.0 - 1.8 * progress

    def entropy_weight(self) -> float:
        progress = min(max((self.epoch - 1) / max(self.total_epochs - 1, 1), 0.0), 1.0)
        return 0.05 * (1.0 - progress) + 0.001

    def probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.selector_logits / self.temperature())

    def selection_values(self) -> torch.Tensor:
        return torch.where(self.selector_logits.detach() >= 0, torch.ones_like(self.selector_logits), -torch.ones_like(self.selector_logits))

    def entropy(self) -> torch.Tensor:
        p = self.probabilities().clamp(1e-6, 1.0 - 1e-6)
        return -(p * p.log() + (1.0 - p) * (1.0 - p).log()).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            generator = torch.Generator(device=x.device).manual_seed(self.seed + self.epoch * 1009 + x.shape[0])
            uniform = torch.rand(x.shape, generator=generator, device=x.device).clamp(1e-6, 1.0 - 1e-6)
            logistic = torch.log(uniform) - torch.log1p(-uniform)
            soft = torch.sigmoid((self.selector_logits.unsqueeze(0) + logistic) / self.temperature())
            hard = (soft >= 0.5).float()
            gate = hard.detach() - soft.detach() + soft
        else:
            soft = self.probabilities().unsqueeze(0)
            gate = (soft >= 0.5).float()
        transformed = gate * x + (1.0 - gate) * stable_reciprocal(x)
        return self.output(self.dd(transformed))
