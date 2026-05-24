#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = "/hpc2hdd/home/rtang906/AgentOCR/outputs/v2_debug/vwa_public_flow_batch_current.json"


def _prepare_env() -> None:
    os.chdir(REPO_ROOT)
    os.environ.setdefault("DATASET", "visualwebarena")
    os.environ.setdefault("REDDIT", "https://example.com")
    os.environ.setdefault("SHOPPING", "https://example.com")
    os.environ.setdefault("WIKIPEDIA", "https://example.com")
    os.environ.setdefault("HOMEPAGE", "https://example.com")
    os.environ.setdefault("CLASSIFIEDS", "https://example.com")
    os.environ.setdefault("CLASSIFIEDS_RESET_TOKEN", "dummy-token")
    os.environ.setdefault("REDDIT_RESET_URL", "https://example.com")


_prepare_env()

from browser_env import ScriptBrowserEnv, create_playwright_action  # noqa: E402


FLOWS: List[Dict[str, Any]] = [
    {
        "name": "example_more",
        "actions": [
            'page.goto("https://www.example.com")',
            'page.get_by_role("link", name="More information...").click()',
        ],
        "expect_url_contains": "iana.org",
    },
    {
        "name": "todomvc_add",
        "actions": [
            'page.goto("https://demo.playwright.dev/todomvc/")',
            'page.get_by_placeholder("What needs to be done?").click()',
            'page.get_by_placeholder("What needs to be done?").fill("hello")',
            'page.get_by_placeholder("What needs to be done?").press("Enter")',
        ],
        "expect_text_contains": "hello",
    },
    {
        "name": "hover_github",
        "actions": [
            'page.goto("https://ianlunn.github.io/Hover/")',
            'page.get_by_role("link", name="Download on GitHub").hover()',
        ],
        "expect_text_contains": "Hover.css",
    },
    {
        "name": "select_option",
        "actions": [
            'page.goto("https://russmaxdesign.github.io/exercise/#link-two")',
            'page.get_by_role("combobox", name="Favourite mammal").select_option("African Wild Dog")',
        ],
        "expect_text_contains": "Favourite mammal",
    },
]


def run_flow(env: ScriptBrowserEnv, flow: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[FLOW] {flow['name']} reset", flush=True)
    obs, info = env.reset()
    env.page.set_default_timeout(5000)
    env.page.set_default_navigation_timeout(5000)
    steps: List[Dict[str, Any]] = [
        {
            "kind": "reset",
            "success": True,
            "url": info["page"].url,
            "obs_keys": sorted(obs.keys()),
        }
    ]
    flow_success = True

    for action_text in flow["actions"]:
        print(f"[FLOW] {flow['name']} action={action_text}", flush=True)
        try:
            obs, success, _, _, info = env.step(create_playwright_action(action_text))
        except Exception as exc:
            steps.append(
                {
                    "kind": "step",
                    "action": action_text,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            flow_success = False
            break
        steps.append(
            {
                "kind": "step",
                "action": action_text,
                "success": bool(success),
                "url": info["page"].url,
            }
        )
        if not success:
            flow_success = False
            break

    final_url = str(steps[-1].get("url", ""))
    final_text = obs.get("text", "") if isinstance(obs, dict) else ""
    if flow_success and "expect_url_contains" in flow:
        flow_success = flow["expect_url_contains"] in final_url
    if flow_success and "expect_text_contains" in flow:
        flow_success = flow["expect_text_contains"] in final_text

    return {
        "name": flow["name"],
        "success": flow_success,
        "final_url": final_url,
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of public VisualWebArena browser flows.")
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--headless", action="store_true", default=False)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = ScriptBrowserEnv(
        observation_type="image_som",
        current_viewport_only=False,
        headless=args.headless,
        sleep_after_execution=0.2,
    )
    results: List[Dict[str, Any]] = []
    try:
        for flow in FLOWS:
            results.append(run_flow(env, flow))
    finally:
        env.close()

    passed = sum(1 for item in results if item["success"])
    payload = {
        "summary": {
            "num_flows": len(results),
            "success_count": passed,
            "success_rate": passed / len(results) if results else 0.0,
        },
        "rows": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote public flow batch to {output_path}")


if __name__ == "__main__":
    main()
