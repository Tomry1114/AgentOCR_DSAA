# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Copyright 2025 verl-agent (GiGPO) Team
# Copyright 2026 AgentOCR Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from importlib import import_module

__all__ = [
    "TrajectoryCollector",
    "adjust_batch",
    "adjust_batchstart",
    "analyze_counterfactual_credit",
    "analyze_trust_rendering",
    "build_step_events",
    "infer_repair_step",
    "run_offline_agentocr_hooks",
]

_IMPORT_MAP = {
    "TrajectoryCollector": (".rollout_loop", "TrajectoryCollector"),
    "adjust_batch": (".utils", "adjust_batch"),
    "adjust_batchstart": (".utils", "adjust_batch"),
    "analyze_counterfactual_credit": (".offline_hooks", "analyze_counterfactual_credit"),
    "analyze_trust_rendering": (".offline_hooks", "analyze_trust_rendering"),
    "build_step_events": (".offline_hooks", "build_step_events"),
    "infer_repair_step": (".offline_hooks", "infer_repair_step"),
    "run_offline_agentocr_hooks": (".offline_hooks", "run_offline_agentocr_hooks"),
}


def __getattr__(name):
    if name not in _IMPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _IMPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
