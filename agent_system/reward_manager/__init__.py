# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Copyright 2025 verl-agent (GiGPO) Team
# Copyright 2026 AgentOCR Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from importlib import import_module

__all__ = [
    "EpisodeRewardManager",
    "EpisodeRewardManager_Compression",
    "CounterfactualCreditConfig",
    "CounterfactualCreditResult",
    "CounterfactualOpticalCreditAssigner",
    "OpticalStepEvent",
]

_IMPORT_MAP = {
    "EpisodeRewardManager": (".episode", "EpisodeRewardManager"),
    "EpisodeRewardManager_Compression": (".episode_with_compression", "EpisodeRewardManager_Compression"),
    "CounterfactualCreditConfig": (".counterfactual_credit", "CounterfactualCreditConfig"),
    "CounterfactualCreditResult": (".counterfactual_credit", "CounterfactualCreditResult"),
    "CounterfactualOpticalCreditAssigner": (".counterfactual_credit", "CounterfactualOpticalCreditAssigner"),
    "OpticalStepEvent": (".counterfactual_credit", "OpticalStepEvent"),
}


def __getattr__(name):
    if name not in _IMPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _IMPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
