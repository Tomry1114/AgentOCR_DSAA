from __future__ import annotations

import importlib.util
from functools import wraps
from pathlib import Path


_UPSTREAM_ENV_PATH = (
    Path(__file__).resolve().parents[3]
    / "agent_system"
    / "environments"
    / "env_package"
    / "webshop"
    / "webshop"
    / "web_agent_site"
    / "envs"
    / "web_agent_text_env.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "_isolated_upstream_webshop_web_agent_text_env",
    _UPSTREAM_ENV_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load upstream WebShop env from {_UPSTREAM_ENV_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
UpstreamWebAgentTextEnv = _MODULE.WebAgentTextEnv
tag_visible = _MODULE.tag_visible


class WebAgentTextEnv(UpstreamWebAgentTextEnv):
    """WebShop text env with result-page titles exposed as clickable actions.

    Upstream WebShop only exposes the product ASIN as the clickable text on the
    search results page. The observation, however, contains the human-readable
    title beside that ASIN. Letting the policy click the visible title removes a
    brittle title->ASIN indirection without adding any oracle information.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._patch_server_item_page()

    @staticmethod
    def _class_list(node) -> list[str]:
        if node is None:
            return []
        classes = node.get("class")
        return list(classes) if classes else []

    @staticmethod
    def _extract_result_titles_from_simple_text(simple_text: str, asin_labels: list[str]) -> dict[str, str]:
        if not simple_text or not asin_labels:
            return {}

        asin_set = set(asin_labels)
        titles_by_asin: dict[str, str] = {}
        segments = [segment.strip() for segment in simple_text.split(" [SEP] ") if segment.strip()]
        skip_tokens = {"back to search", "next >", "< prev"}

        for idx, segment in enumerate(segments[:-1]):
            lowered = segment.lower()
            if lowered not in asin_set:
                continue

            for next_segment in segments[idx + 1 :]:
                candidate = next_segment.strip()
                lowered_candidate = candidate.lower()
                if not candidate:
                    continue
                if lowered_candidate in asin_set:
                    break
                if lowered_candidate in skip_tokens or lowered_candidate.startswith("page "):
                    continue
                if candidate.startswith("$"):
                    continue
                titles_by_asin[lowered] = lowered_candidate
                break

        return titles_by_asin

    def _patch_server_item_page(self):
        current = getattr(self.server, "item_page")
        if getattr(current, "_agentocr_title_click_patch", False):
            return

        @wraps(current)
        def wrapped_item_page(session_id, **kwargs):
            clickable_name = kwargs.get("clickable_name")
            text_to_clickable = kwargs.get("text_to_clickable") or {}
            clickable = text_to_clickable.get(clickable_name)
            if clickable is not None:
                clickable_classes = clickable.get("class") or []
                if clickable_classes and clickable_classes[0] == "product-link":
                    asin_label = clickable.get_text(strip=True).lower()
                    if asin_label and clickable_name != asin_label:
                        kwargs = dict(kwargs)
                        kwargs["clickable_name"] = asin_label
            return current(session_id, **kwargs)

        wrapped_item_page._agentocr_title_click_patch = True  # type: ignore[attr-defined]
        self.server.item_page = wrapped_item_page

    def _should_hide_result_nav_button(self, label: str, product_links=None) -> bool:
        current_url = str(self.state.get("url", "") or "").lower()
        on_results_page = "search_results" in current_url
        if not on_results_page:
            return False

        session = self.server.user_sessions.get(self.session) or {}
        has_seen_result_product = bool(session.get("asins"))
        if has_seen_result_product:
            return False

        if product_links is None:
            html_obj = self._parse_html()
            product_links = html_obj.find_all(class_="product-link")

        return bool(product_links) and label in {"back to search", "next >", "< prev"}

    def convert_html_to_text(self, html, simple=False):
        html_obj = self._parse_html(html)
        texts = html_obj.findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        product_links = html_obj.find_all(class_="product-link")
        product_title_index = 0
        if simple:
            processed_segments = []
            for t in visible_texts:
                if t == "\n":
                    continue

                parent = t.parent
                parent_classes = self._class_list(parent)
                stripped_text = t.strip()
                if not stripped_text:
                    continue

                if parent.name == "button":
                    if self._should_hide_result_nav_button(stripped_text.lower(), product_links):
                        continue
                    processed_segments.append(f"[button] {stripped_text} [button_]")
                elif parent.name == "label":
                    if f'"{t}"' in self.state["url"]:
                        processed_segments.append(f"[clicked button] {stripped_text} [clicked button_]")
                    else:
                        processed_segments.append(f"[button] {stripped_text} [button_]")
                elif "product-link" in parent_classes:
                    continue
                elif "product-title" in parent_classes:
                    product_title_index += 1
                    processed_segments.append(
                        f"[button] [item {product_title_index}] {stripped_text} [button_]"
                    )
                else:
                    processed_segments.append(stripped_text)
            return " [SEP] ".join(processed_segments)

        observation = ""
        for t in visible_texts:
            if t == "\n":
                continue

            parent = t.parent
            parent_classes = self._class_list(parent)
            stripped_text = t.strip()

            if parent.name == "button":
                if self._should_hide_result_nav_button(stripped_text.lower(), product_links):
                    continue
                processed_t = f"[button] {t} [button_]"
            elif parent.name == "label":
                if f'"{t}"' in self.state["url"]:
                    processed_t = f"  [clicked button] {t} [clicked button_]"
                    observation = f"You have clicked {t}.\n" + observation
                else:
                    processed_t = f"  [button] {t} [button_]"
            elif "product-link" in parent_classes:
                continue
            elif "product-title" in parent_classes and stripped_text:
                product_title_index += 1
                processed_t = f"\n[button] [item {product_title_index}] {stripped_text} [button_]"
            else:
                processed_t = str(t)

            observation += processed_t + "\n"

        return observation

    def get_available_actions(self):
        html_obj = self._parse_html()
        goal = (self.server.user_sessions.get(self.session) or {}).get("goal") or {}
        session = self.server.user_sessions.get(self.session) or {}

        search_bar = html_obj.find(id="search_input")
        has_search_bar = search_bar is not None
        buttons = html_obj.find_all(class_="btn")
        product_links = html_obj.find_all(class_="product-link")
        buying_options = html_obj.select('input[type="radio"]')
        simple_text = super().convert_html_to_text(self.state["html"], simple=True)
        asin_labels = [link.get_text(strip=True).lower() for link in product_links if link.get_text(strip=True)]
        titles_by_asin = self._extract_result_titles_from_simple_text(simple_text, asin_labels)
        title_counts = {}
        for title_label in titles_by_asin.values():
            title_counts[title_label] = title_counts.get(title_label, 0) + 1

        self.text_to_clickable = {}
        visible_clickables = []
        deferred_buttons = []

        for button in buttons:
            label = button.get_text(strip=True).lower()
            if label == "search":
                # Upstream exposes a Search button as clickable text, but the
                # environment does not execute click[search]; only search[...] is
                # a real admissible action on the search page.
                continue
            if self._should_hide_result_nav_button(label, product_links):
                # Force at least one product-page inspection before allowing
                # result-page navigation that often derails the policy.
                continue
            if label:
                deferred_buttons.append((label, button))

        for item_index, link in enumerate(product_links, start=1):
            asin_label = link.get_text(strip=True).lower()
            alias_label = f"item {item_index}"
            if asin_label:
                self.text_to_clickable[asin_label] = link
            self.text_to_clickable[alias_label] = link

            title_label = titles_by_asin.get(asin_label, "")
            if title_label and title_counts.get(title_label, 0) == 1:
                self.text_to_clickable[title_label] = link
            visible_clickables.append(alias_label)

        for opt in buying_options:
            opt_value = opt.get("value")
            if opt_value:
                lowered_value = f"{opt_value}".lower()
                self.text_to_clickable[lowered_value] = opt
                visible_clickables.append(lowered_value)

        for label, button in deferred_buttons:
            self.text_to_clickable[label] = button
            visible_clickables.append(label)

        return dict(
            has_search_bar=has_search_bar,
            clickables=visible_clickables,
            goal_query=str(goal.get("query", "") or ""),
            goal_attributes=[str(item) for item in (goal.get("attributes") or []) if item],
            goal_instruction_text=str(goal.get("instruction_text", "") or ""),
            goal_options={
                str(key): str(value)
                for key, value in (goal.get("goal_options") or {}).items()
                if str(value).strip()
            },
        )
