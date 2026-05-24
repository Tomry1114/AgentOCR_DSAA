#!/usr/bin/env python3
import os
import runpy
import sys


def _sanitize_sys_path(project_root: str) -> None:
    cleaned = []
    delayed = []
    normalized_root = os.path.realpath(project_root)

    for entry in sys.path:
        normalized = os.path.realpath(entry or os.getcwd())
        if normalized == normalized_root:
            delayed.append(entry)
        else:
            cleaned.append(entry)

    if project_root not in cleaned:
        cleaned.append(project_root)
    for entry in delayed:
        if entry not in cleaned:
            cleaned.append(entry)
    sys.path[:] = cleaned


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_module_sanitized.py <module> [args...]")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sanitize_sys_path(project_root)

    module_name = sys.argv[1]
    sys.argv = [module_name] + sys.argv[2:]

    if os.environ.get("AGENTOCR_IMPORT_DEBUG") == "1":
        print("[AGENTOCR_IMPORT_DEBUG] sys.path=", file=sys.stderr)
        for idx, entry in enumerate(sys.path):
            print(f"  {idx}: {entry}", file=sys.stderr)

    runpy.run_module(module_name, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
