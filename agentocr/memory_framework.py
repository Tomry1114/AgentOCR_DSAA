from __future__ import annotations

from dataclasses import dataclass


EMA_COMPONENT = "Evolvable Memory Abstractions"
EGRC_COMPONENT = "Evidence-Grounded Retention Control"
FMR_COMPONENT = "Failure-Driven Memory Revision"


@dataclass(frozen=True)
class MemoryFrameworkConfig:
    enable_ema: bool = True
    enable_egrc: bool = True
    enable_fmr: bool = True


DEFAULT_MEMORY_FRAMEWORK_CONFIG = MemoryFrameworkConfig()


def resolve_memory_framework_config(
    config: MemoryFrameworkConfig | None,
) -> MemoryFrameworkConfig:
    return config if config is not None else DEFAULT_MEMORY_FRAMEWORK_CONFIG
