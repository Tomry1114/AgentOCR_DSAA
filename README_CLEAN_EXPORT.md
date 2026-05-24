This directory is a cleaned export of the AgentOCR codebase for sharing or reuse.

Included:
- core source code such as `agentocr/`, `agent_system/`, `verl/`, `tools/`, `scripts/`
- training and inference shell scripts
- configs and packaging files such as `configs/`, `requirements*.txt`, `setup.py`, `pyproject.toml`
- external/runtime code that the current project depends on

Excluded:
- model weights and wheel artifacts
- datasets and generated outputs
- checkpoints, wandb runs, diagnostics, and logs
- paper-writing materials and research notes
- supplementary experiment artifacts and temporary debug data

Repository-specific exclusions applied when creating this export:
- `Paper/`
- `wandb/`
- `checkpoints/`
- `data/`
- `outputs/`
- `diagnostics/`
- `logs*/`, `log*/`
- `refine-logs/`, `idea-stage/`, `review-stage/`
- `supplement-data/`, `supplement-logs/`
- `PROMPT_TEMPLATES.md`
- `DEVELOPMENT_LOG.md`
- `RESEARCH_REVIEW_AGENTOCR.md`
- `findings.md`

Note:
- This is a code-only cleaned snapshot, not a guaranteed standalone release package.
- Paths inside scripts may still reference your original filesystem layout.
