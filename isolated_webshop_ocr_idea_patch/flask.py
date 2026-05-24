from __future__ import annotations

from contextlib import nullcontext
from jinja2 import Template


class Flask:
    def __init__(self, import_name: str):
        self.import_name = import_name

    def route(self, _rule: str, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    def app_context(self):
        return nullcontext()

    def test_request_context(self):
        return nullcontext()


def render_template_string(template: str, **kwargs) -> str:
    return Template(template).render(**kwargs)
