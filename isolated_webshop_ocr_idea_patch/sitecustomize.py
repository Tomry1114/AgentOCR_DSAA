"""Process-local patch hook for isolated WebShop OCR experiments.

This file is imported automatically by Python when its parent directory is
present on ``PYTHONPATH``. The patch is gated by
``AGENTOCR_ENABLE_WEBSHOP_OCR_PATCH=1`` so existing experiments remain
unaffected unless a dedicated launcher opts in.
"""

from __future__ import annotations

import os
import sys


def _env_enabled() -> bool:
    return os.environ.get("AGENTOCR_ENABLE_WEBSHOP_OCR_PATCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _clear_incompatible_allocator_env() -> None:
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if alloc_conf and "expandable_segments:True" in alloc_conf:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


def _patch_make_envs() -> None:
    import agent_system.environments as environments_module
    import agent_system.environments.env_manager as env_manager_module

    current_make_envs = getattr(env_manager_module, "make_envs", None)
    if current_make_envs is None:
        return
    if getattr(current_make_envs, "_agentocr_webshop_ocr_patch", False):
        return

    original_make_envs = current_make_envs

    def _is_webshop_ocr_env(config) -> bool:
        env_name = str(getattr(config.env, "env_name", "") or "").strip().lower()
        return env_name in {"webshopocr", "webshopocridea", "webshop_ocr", "webshop_ocr_idea"}

    def patched_make_envs(config):
        if _is_webshop_ocr_env(config):
            from isolated_webshop_ocr_idea.envs import make_webshop_ocr_envs

            return make_webshop_ocr_envs(config)
        return original_make_envs(config)

    patched_make_envs._agentocr_webshop_ocr_patch = True  # type: ignore[attr-defined]
    env_manager_module.make_envs = patched_make_envs
    environments_module.make_envs = patched_make_envs


def _patch_ray_worker_env_passthrough() -> None:
    import verl.single_controller.ray.base as ray_base_module

    ray_class = getattr(ray_base_module, "RayClassWithInitArgs", None)
    if ray_class is None:
        return

    current_update_options = getattr(ray_class, "update_options", None)
    if current_update_options is None:
        return
    if getattr(current_update_options, "_agentocr_webshop_env_patch", False):
        return

    passthrough_keys = (
        "VLLM_ATTENTION_BACKEND",
        "VLLM_USE_FLASHINFER_SAMPLER",
        "VLLM_DISABLE_FLASHINFER_PREFILL",
        "VLLM_USE_V1",
        "VERL_HF_ATTN_IMPLEMENTATION",
    )

    def patched_update_options(self, options):
        runtime_env = options.get("runtime_env")
        if isinstance(runtime_env, dict):
            env_vars = dict(runtime_env.get("env_vars", {}) or {})
            for key in passthrough_keys:
                value = os.environ.get(key)
                if value is not None and key not in env_vars:
                    env_vars[key] = value
            options = dict(options)
            options["runtime_env"] = dict(runtime_env)
            options["runtime_env"]["env_vars"] = env_vars
        return current_update_options(self, options)

    patched_update_options._agentocr_webshop_env_patch = True  # type: ignore[attr-defined]
    ray_class.update_options = patched_update_options


def _patch_disable_upstream_flash_attn() -> None:
    import vllm.attention.layer as attention_layer_module

    current_check = getattr(attention_layer_module,
                            "check_upstream_fa_availability", None)
    if current_check is None:
        return
    if getattr(current_check, "_agentocr_webshop_disable_upstream_fa", False):
        return

    def patched_check_upstream_fa_availability(dtype):
        return False

    patched_check_upstream_fa_availability._agentocr_webshop_disable_upstream_fa = True  # type: ignore[attr-defined]
    attention_layer_module.check_upstream_fa_availability = patched_check_upstream_fa_availability

    # If Qwen VL modules were already imported in this interpreter, replace
    # their local symbol binding as well so they cannot force-switch to the
    # broken upstream flash-attn package on this node.
    for module_name in (
            "vllm.model_executor.models.qwen2_5_vl",
            "vllm.model_executor.models.qwen3_vl",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module,
                                          "check_upstream_fa_availability"):
            setattr(module, "check_upstream_fa_availability",
                    patched_check_upstream_fa_availability)


def _patch_ocrtool_qwen3_prefix_signature() -> None:
    import inspect

    from agentocr.ocrtool import OCRTool

    current_method = getattr(OCRTool, "_qwen3_page_prefix_lines", None)
    if current_method is None:
        return
    if getattr(current_method, "_agentocr_webshop_snapshot_compat", False):
        return

    signature = inspect.signature(current_method)
    if "snapshot" in signature.parameters:
        return

    def patched_qwen3_page_prefix_lines(
        self,
        header,
        page_index,
        snapshot_lines=None,
        *,
        snapshot=None,
        **_,
    ):
        if snapshot_lines is None:
            snapshot_lines = snapshot if snapshot is not None else []
        return current_method(self, header, page_index, snapshot_lines)

    patched_qwen3_page_prefix_lines._agentocr_webshop_snapshot_compat = True  # type: ignore[attr-defined]
    OCRTool._qwen3_page_prefix_lines = patched_qwen3_page_prefix_lines


if _env_enabled():
    _clear_incompatible_allocator_env()
    _patch_make_envs()
    _patch_ray_worker_env_passthrough()
    _patch_disable_upstream_flash_attn()
    _patch_ocrtool_qwen3_prefix_signature()
