# Copyright 2026 AgentOCR Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from importlib import import_module

__all__ = [
    "BaseOCRTool",
    "OCRTool",
    "SegmentCache",
    "split_into_segments",
    "RenderDecision",
    "SegmentTrustMetadata",
    "TrustCalibratedRenderPolicy",
    "TrustPolicyConfig",
]

_IMPORT_MAP = {
    "BaseOCRTool": (".base", "BaseOCRTool"),
    "OCRTool": (".ocrtool", "OCRTool"),
    "SegmentCache": (".ocrtool", "SegmentCache"),
    "split_into_segments": (".ocrtool", "split_into_segments"),
    "RenderDecision": (".trust_policy", "RenderDecision"),
    "SegmentTrustMetadata": (".trust_policy", "SegmentTrustMetadata"),
    "TrustCalibratedRenderPolicy": (".trust_policy", "TrustCalibratedRenderPolicy"),
    "TrustPolicyConfig": (".trust_policy", "TrustPolicyConfig"),
}


def __getattr__(name):
    if name not in _IMPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _IMPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
