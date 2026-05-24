# Copyright 2026 AgentOCR Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class OpticalStepEvent:
    """Minimal per-step information needed for optical credit assignment."""

    step: int
    action_valid: bool = True
    compression_factor: float = 1.0
    visual_token_count: float = 0.0
    segment_count: int = 0


@dataclass(frozen=True)
class CounterfactualCreditConfig:
    """Coefficients for dense counterfactual optical credit."""

    success_reward: float = 1.0
    terminal_failure_penalty: float = -1.0
    invalid_action_penalty: float = -0.2
    success_compression_bonus: float = 0.01
    failure_compression_blame: float = 0.03
    repair_decay: float = 0.65
    max_compression_factor: float = 5.0


@dataclass(frozen=True)
class CounterfactualCreditResult:
    dense_rewards: List[float]
    blame_scores: List[float]
    repair_step: Optional[int]
    summary: str


class CounterfactualOpticalCreditAssigner:
    """Assign dense rewards from success/failure and a counterfactual repair point.

    The class is a lightweight prototype for the "Counterfactual Optical Credit
    Assignment" idea. It does not replace the existing reward manager yet; it
    exposes a small, testable primitive that can later be called from rollout
    post-processing once repaired/counterfactual traces are available.
    """

    def __init__(self, config: Optional[CounterfactualCreditConfig] = None):
        self.config = config or CounterfactualCreditConfig()

    def assign(
        self,
        events: Iterable[OpticalStepEvent],
        is_success: bool,
        repair_step: Optional[int] = None,
    ) -> CounterfactualCreditResult:
        ordered_events = sorted(events, key=lambda event: event.step)
        if not ordered_events:
            return CounterfactualCreditResult([], [], repair_step, "empty trajectory")

        dense_rewards = [0.0 for _ in ordered_events]
        blame_scores = [0.0 for _ in ordered_events]

        for index, event in enumerate(ordered_events):
            if not event.action_valid:
                dense_rewards[index] += self.config.invalid_action_penalty

            compression = self._safe_compression(event.compression_factor)
            if is_success:
                dense_rewards[index] += log(compression) * self.config.success_compression_bonus
            elif repair_step is not None:
                distance = abs(event.step - repair_step)
                blame = self.config.repair_decay**distance
                if event.step < repair_step:
                    blame *= 0.5
                blame_scores[index] = blame
                dense_rewards[index] -= log(compression) * self.config.failure_compression_blame * blame

        if is_success:
            dense_rewards[-1] += self.config.success_reward
            summary = "success: compression receives small positive credit"
        else:
            dense_rewards[-1] += self.config.terminal_failure_penalty
            if repair_step is None:
                blame_scores[-1] = 1.0
                summary = "failure: no repair step, terminal blame only"
            else:
                summary = "failure: blame concentrated around counterfactual repair step"

        return CounterfactualCreditResult(
            dense_rewards=dense_rewards,
            blame_scores=blame_scores,
            repair_step=repair_step,
            summary=summary,
        )

    def _safe_compression(self, compression_factor: float) -> float:
        value = max(1.0, float(compression_factor))
        return min(value, self.config.max_compression_factor)
