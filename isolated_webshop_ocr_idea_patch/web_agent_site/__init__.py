from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_repo_web_agent_site = (
    Path(__file__).resolve().parents[2]
    / "agent_system"
    / "environments"
    / "env_package"
    / "webshop"
    / "webshop"
    / "web_agent_site"
)
if _repo_web_agent_site.exists():
    repo_path = str(_repo_web_agent_site)
    if repo_path not in __path__:
        __path__.append(repo_path)
