from pkgutil import extend_path

from gym.envs.registration import register

__path__ = extend_path(__path__, __name__)

from .web_agent_text_env import WebAgentTextEnv

register(
    id="WebAgentTextEnv-v0",
    entry_point="web_agent_site.envs:WebAgentTextEnv",
)

__all__ = ["WebAgentTextEnv"]
