from __future__ import annotations

import json
import math
import os
import random
import re
from urllib.parse import urlencode
from ast import literal_eval
from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any

from jinja2 import Template

from web_agent_site.utils import BASE_DIR, DEFAULT_ATTR_PATH, DEFAULT_FILE_PATH

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

SEARCH_RETURN_N = 50
PRODUCT_WINDOW = 10

END_BUTTON = "Buy Now"
NEXT_PAGE = "Next >"
PREV_PAGE = "< Prev"
BACK_TO_SEARCH = "Back to Search"

ACTION_TO_TEMPLATE = {
    "Description": "description_page.html",
    "Features": "features_page.html",
    "Reviews": "review_page.html",
    "Attributes": "attributes_page.html",
}

_LAST_ALL_PRODUCTS: list[dict[str, Any]] = []
_SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "can",
    "dollars",
    "find",
    "for",
    "i",
    "looking",
    "lower",
    "me",
    "price",
    "than",
    "the",
    "with",
    "you",
}
_QUERY_INTRO_RE = re.compile(
    r"^\s*(?:find me|i am looking for|i'm looking for|looking for|can you find me|can you find)\s+",
    re.IGNORECASE,
)
_PRICE_CLAUSE_RE = re.compile(
    r"(?:,?\s*and\s*)?price\s+lower\s+than\s+[\d.]+\s+dollars\b",
    re.IGNORECASE,
)
_OPTION_LABELS = (
    "color",
    "size",
    "fit type",
    "style name",
    "material type",
    "special size type",
)
_OPTION_CLAUSE_RE = re.compile(
    r"\b(" + "|".join(re.escape(label) for label in _OPTION_LABELS) + r")\s*:\s*([^,]+)",
    re.IGNORECASE,
)
_PRODUCT_TYPE_PATTERNS = (
    ("dress_shirt", (r"\bdress shirts?\b", r"\bbutton[- ]down shirts?\b", r"\btuxedo shirts?\b", r"\bformal shirts?\b")),
    ("t_shirt", (r"\bt[- ]shirts?\b", r"\btees?\b")),
    ("tank_top", (r"\btank tops?\b", r"\btanks\b", r"\bcamis?\b")),
    ("henley", (r"\bhenleys?\b",)),
    ("polo", (r"\bpolos?\b", r"\bpolo shirts?\b")),
    ("shorts", (r"\bshorts\b",)),
    ("pants", (r"\bpants\b", r"\btrousers\b", r"\bjoggers?\b", r"\bcapris?\b")),
    ("blanket", (r"\bblankets?\b", r"\bthrows?\b")),
    ("coat_jacket", (r"\bcoats?\b", r"\bjackets?\b", r"\boutwear\b")),
    ("shoes", (r"\bshoes\b", r"\bslippers?\b", r"\bsandals?\b", r"\bboots?\b")),
)
_GENDER_PATTERNS = (
    ("women", (r"\bwomen'?s\b", r"\bwomens\b", r"\bladies\b", r"\bgirls?\b")),
    ("men", (r"\bmen'?s\b", r"\bmens\b", r"\bboys?\b")),
)


def _render_template_string(template: str, **kwargs) -> str:
    render_kwargs = dict(kwargs)
    render_kwargs.setdefault("url_for", _url_for)
    render_kwargs.setdefault("dict", dict)
    return Template(template).render(**render_kwargs)


def _url_for(endpoint: str, **kwargs) -> str:
    if endpoint == "static":
        filename = kwargs.get("filename", "")
        return f"/static/{filename}"
    if kwargs:
        return f"/{endpoint}?{urlencode(kwargs, doseq=True)}"
    return f"/{endpoint}"


def map_action_to_html(action, **kwargs):
    action_name, action_arg = parse_action(action)
    if action_name == "start":
        path = os.path.join(TEMPLATE_DIR, "search_page.html")
        html = _render_template_string(
            read_html_template(path=path),
            session_id=kwargs["session_id"],
            instruction_text=kwargs["instruction_text"],
        )
    elif action_name == "search":
        path = os.path.join(TEMPLATE_DIR, "results_page.html")
        html = _render_template_string(
            read_html_template(path=path),
            session_id=kwargs["session_id"],
            products=kwargs["products"],
            keywords=kwargs["keywords"],
            page=kwargs["page"],
            total=kwargs["total"],
            instruction_text=kwargs["instruction_text"],
        )
    elif action_name == "click" and action_arg == END_BUTTON:
        path = os.path.join(TEMPLATE_DIR, "done_page.html")
        html = _render_template_string(
            read_html_template(path),
            session_id=kwargs["session_id"],
            reward=kwargs["reward"],
            asin=kwargs["asin"],
            options=kwargs["options"],
            reward_info=kwargs.get("reward_info"),
            goal_attrs=kwargs.get("goal_attrs"),
            purchased_attrs=kwargs.get("purchased_attrs"),
            goal=kwargs.get("goal"),
            mturk_code=kwargs.get("mturk_code"),
            query=kwargs.get("query"),
            category=kwargs.get("category"),
            product_category=kwargs.get("product_category"),
        )
    elif action_name == "click" and action_arg in ACTION_TO_TEMPLATE:
        path = os.path.join(TEMPLATE_DIR, ACTION_TO_TEMPLATE[action_arg])
        html = _render_template_string(
            read_html_template(path),
            session_id=kwargs["session_id"],
            product_info=kwargs["product_info"],
            keywords=kwargs["keywords"],
            page=kwargs["page"],
            asin=kwargs["asin"],
            options=kwargs["options"],
            instruction_text=kwargs.get("instruction_text"),
        )
    elif action_name == "click":
        path = os.path.join(TEMPLATE_DIR, "item_page.html")
        html = _render_template_string(
            read_html_template(path),
            session_id=kwargs["session_id"],
            product_info=kwargs["product_info"],
            keywords=kwargs["keywords"],
            page=kwargs["page"],
            asin=kwargs["asin"],
            options=kwargs["options"],
            instruction_text=kwargs.get("instruction_text"),
            show_attrs=kwargs["show_attrs"],
        )
    else:
        raise ValueError("Action name not recognized.")
    return html


def read_html_template(path):
    with open(path) as f:
        template = f.read()
    return template


def parse_action(action):
    pattern = re.compile(r"(.+)\[(.+)\]")
    m = re.match(pattern, action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()
    return action_name, action_arg


def convert_web_app_string_to_var(name, string):
    if name == "keywords":
        keywords = string
        if keywords.startswith("["):
            keywords = literal_eval(keywords)
        else:
            keywords = [keywords]
        var = keywords
    elif name == "page":
        page = int(string)
        var = page
    else:
        raise ValueError("Name of variable not recognized.")
    return var


class _SimpleDoc:
    def __init__(self, asin: str):
        self._asin = asin

    def raw(self):
        return json.dumps({"id": self._asin})


class _SimpleHit:
    def __init__(self, docid: str):
        self.docid = docid


def _tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    if not drop_stopwords:
        return tokens
    return [token for token in tokens if token not in _SEARCH_STOPWORDS]


def _canonical_phrase(text: str) -> str:
    return " ".join(_tokenize(text, drop_stopwords=False))


def _normalize_query_for_search(text: str) -> str:
    normalized = str(text).lower()
    normalized = _QUERY_INTRO_RE.sub("", normalized)
    normalized = _PRICE_CLAUSE_RE.sub("", normalized)
    for label in _OPTION_LABELS:
        normalized = re.sub(rf"\b{re.escape(label)}\s*:\s*", "", normalized)
    normalized = normalized.replace("|", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,")
    return normalized


def _extract_query_option_values(text: str) -> set[str]:
    values: set[str] = set()
    for _, raw_value in _OPTION_CLAUSE_RE.findall(str(text).lower()):
        canonical = _canonical_phrase(raw_value)
        if canonical:
            values.add(canonical)
    return values


def _collect_product_option_values(product: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for option_values in (product.get("options") or {}).values():
        for option_value in option_values or []:
            canonical = _canonical_phrase(option_value)
            if canonical:
                values.add(canonical)
    return values


def _collect_instruction_texts(product: dict[str, Any]) -> list[str]:
    texts: list[str] = []

    instruction_text = product.get("instruction_text")
    if instruction_text:
        texts.append(str(instruction_text))

    instruction_attributes = product.get("instruction_attributes") or []
    if instruction_attributes:
        texts.append(" ".join(str(item) for item in instruction_attributes if item))

    human_instructions = product.get("instructions") or []
    for instruction in human_instructions:
        raw_text = instruction.get("instruction")
        if raw_text:
            texts.append(str(raw_text))
        raw_attrs = instruction.get("instruction_attributes") or []
        if raw_attrs:
            texts.append(" ".join(str(item) for item in raw_attrs if item))
        raw_options = instruction.get("instruction_options") or []
        if raw_options:
            texts.append(" ".join(str(item) for item in raw_options if item))

    return texts


def _detect_product_types(text: str) -> set[str]:
    lowered = str(text or "").lower()
    detected: set[str] = set()
    for type_name, patterns in _PRODUCT_TYPE_PATTERNS:
        if any(re.search(pattern, lowered) for pattern in patterns):
            detected.add(type_name)
    return detected


def _detect_gender_label(text: str) -> str:
    lowered = str(text or "").lower()
    detected = [
        label
        for label, patterns in _GENDER_PATTERNS
        if any(re.search(pattern, lowered) for pattern in patterns)
    ]
    if len(detected) == 1:
        return detected[0]
    return ""


class SimpleLexicalSearcher:
    _BM25_K1 = 1.5
    _BM25_B = 0.75

    def __init__(self, products: list[dict[str, Any]]):
        self._docs: list[dict[str, Any]] = []
        self._doc_map: dict[str, _SimpleDoc] = {}
        self._doc_freqs: Counter[str] = Counter()
        total_doc_len = 0

        for product in products:
            asin = product["asin"]
            option_values = _collect_product_option_values(product)
            attribute_text = " ".join(product.get("Attributes", []) or [])
            option_text = " ".join(sorted(option_values))
            text_fields = [
                product.get("Title", ""),
                product.get("Description", ""),
                " ".join(product.get("BulletPoints", []) or []),
                attribute_text,
                attribute_text,
                product.get("category", ""),
                product.get("product_category", ""),
                option_text,
                option_text,
            ]
            tokens = _tokenize(" ".join(str(x) for x in text_fields if x))
            token_counts = Counter(tokens)
            unique_tokens = set(token_counts)
            self._doc_freqs.update(unique_tokens)
            total_doc_len += len(tokens)

            category_core = _normalize_query_for_search(product.get("product_category", ""))
            title_category_text = " ".join(
                str(product.get(field, ""))
                for field in ("Title", "product_category", "category", "Attributes")
            )
            self._docs.append(
                {
                    "asin": asin,
                    "token_counts": token_counts,
                    "token_set": unique_tokens,
                    "doc_len": len(tokens),
                    "option_values": option_values,
                    "category_tokens": set(_tokenize(category_core)),
                    "doc_types": _detect_product_types(title_category_text),
                    "doc_gender": _detect_gender_label(title_category_text),
                }
            )
            self._doc_map[asin] = _SimpleDoc(asin)

        self._num_docs = len(self._docs)
        self._avg_doc_len = (total_doc_len / self._num_docs) if self._num_docs else 0.0

    def _bm25_score(self, query_tokens: list[str], doc: dict[str, Any]) -> float:
        if not query_tokens or not doc["doc_len"] or self._avg_doc_len <= 0:
            return 0.0

        token_counts: Counter[str] = doc["token_counts"]
        score = 0.0
        doc_len = doc["doc_len"]
        denom_norm = self._BM25_K1 * (1 - self._BM25_B + self._BM25_B * (doc_len / self._avg_doc_len))
        for token in query_tokens:
            freq = token_counts.get(token, 0)
            if freq <= 0:
                continue
            doc_freq = self._doc_freqs.get(token, 0)
            idf = math.log(1 + (self._num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            score += idf * ((freq * (self._BM25_K1 + 1)) / (freq + denom_norm))
        return score

    def search(self, query: str, k: int = SEARCH_RETURN_N):
        normalized_query = _normalize_query_for_search(query)
        query_tokens = _tokenize(normalized_query)
        query_token_set = set(query_tokens)
        query_option_values = _extract_query_option_values(query)
        query_types = _detect_product_types(normalized_query)
        query_gender = _detect_gender_label(normalized_query)
        scored: list[tuple[float, str]] = []
        for doc in self._docs:
            token_overlap = len(query_token_set & doc["token_set"])
            score = self._bm25_score(query_tokens, doc) + 0.1 * token_overlap

            category_overlap = len(query_token_set & doc["category_tokens"])
            score += 0.5 * category_overlap
            if query_gender and doc["doc_gender"]:
                score += 2.0 if query_gender == doc["doc_gender"] else -3.0

            doc_types = doc["doc_types"]
            if query_types:
                shared_types = query_types & doc_types
                if shared_types:
                    score += 6.0 * len(shared_types)
                elif doc_types:
                    score -= 4.0
                if "dress_shirt" in query_types and {"t_shirt", "tank_top"} & doc_types:
                    score -= 6.0
                if "shorts" in query_types and {"coat_jacket", "pants"} & doc_types:
                    score -= 6.0
                if "blanket" in query_types and {"coat_jacket", "pants", "shoes"} & doc_types:
                    score -= 6.0

            score += 3.0 * sum(
                1
                for option_value in query_option_values
                if option_value in doc["option_values"]
            )

            if score > 0:
                scored.append((score, doc["asin"]))
        if not scored:
            fallback = [doc["asin"] for doc in self._docs[:k]]
            return [_SimpleHit(asin) for asin in fallback]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [_SimpleHit(asin) for _, asin in scored[:k]]

    def doc(self, docid: str):
        return self._doc_map[docid]


def get_top_n_product_from_keywords(
    keywords,
    search_engine,
    all_products,
    product_item_dict,
    attribute_to_asins=None,
):
    if keywords[0] == "<r>":
        top_n_products = random.sample(all_products, k=min(SEARCH_RETURN_N, len(all_products)))
    elif keywords[0] == "<a>":
        attribute = " ".join(keywords[1:]).strip()
        asins = attribute_to_asins[attribute]
        top_n_products = [p for p in all_products if p["asin"] in asins]
    elif keywords[0] == "<c>":
        category = keywords[1].strip()
        top_n_products = [p for p in all_products if p["category"] == category]
    elif keywords[0] == "<q>":
        query = " ".join(keywords[1:]).strip()
        top_n_products = [p for p in all_products if p["query"] == query]
    else:
        keywords = " ".join(keywords)
        hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
        docs = [search_engine.doc(hit.docid) for hit in hits]
        top_n_asins = [json.loads(doc.raw())["id"] for doc in docs]
        top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
    return top_n_products


def get_product_per_page(top_n_products, page):
    return top_n_products[(page - 1) * PRODUCT_WINDOW : page * PRODUCT_WINDOW]


def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product["asin"]
        pricing = product["pricing"]
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[asin] = price
    return product_prices


def init_search_engine(num_products=None):
    del num_products
    return SimpleLexicalSearcher(_LAST_ALL_PRODUCTS)


def clean_product_keys(products):
    for product in products:
        product.pop("product_information", None)
        product.pop("brand", None)
        product.pop("brand_url", None)
        product.pop("list_price", None)
        product.pop("availability_quantity", None)
        product.pop("availability_status", None)
        product.pop("total_reviews", None)
        product.pop("total_answered_questions", None)
        product.pop("seller_id", None)
        product.pop("seller_name", None)
        product.pop("fulfilled_by_amazon", None)
        product.pop("fast_track_message", None)
        product.pop("aplus_present", None)
        product.pop("small_description_old", None)
    return products


def load_products(filepath=DEFAULT_FILE_PATH, attrpath=DEFAULT_ATTR_PATH, num_products=None, human_goals=True):
    global _LAST_ALL_PRODUCTS

    human_attr_path = os.environ.get("WEBSHOP_HUMAN_ATTR_PATH")
    if not human_attr_path:
        human_attr_path = os.path.join(os.path.dirname(filepath), "items_human_ins.json")

    with open(filepath) as f:
        products = json.load(f)
    products = clean_product_keys(products)

    all_reviews = dict()
    all_ratings = dict()
    human_attributes = {}
    if human_goals:
        with open(human_attr_path) as f:
            human_attributes = json.load(f)
    with open(attrpath) as f:
        attributes = json.load(f)

    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    if num_products is not None:
        products = products[:num_products]
    for i, p in enumerate(products):
        asin = p["asin"]
        if asin == "nan" or len(asin) > 10:
            continue
        if asin in asins:
            continue
        asins.add(asin)

        products[i]["category"] = p["category"]
        products[i]["query"] = p["query"]
        products[i]["product_category"] = p["product_category"]
        products[i]["Title"] = p["name"]
        products[i]["Description"] = p["full_description"]
        products[i]["Reviews"] = all_reviews.get(asin, [])
        products[i]["Rating"] = all_ratings.get(asin, "N.A.")
        for r in products[i]["Reviews"]:
            if "score" not in r:
                r["score"] = r.pop("stars")
            if "review" not in r:
                r["body"] = ""
            else:
                r["body"] = r.pop("review")
        products[i]["BulletPoints"] = p["small_description"] if isinstance(p["small_description"], list) else [p["small_description"]]

        pricing = p.get("pricing")
        if pricing is None or not pricing:
            pricing = [100.0]
            price_tag = "$100.0"
        else:
            pricing = [float(Decimal(re.sub(r"[^\d.]", "", price))) for price in pricing.split("$")[1:]]
            if len(pricing) == 1:
                price_tag = f"${pricing[0]}"
            else:
                price_tag = f"${pricing[0]} to ${pricing[1]}"
                pricing = pricing[:2]
        products[i]["pricing"] = pricing
        products[i]["Price"] = price_tag

        options = dict()
        customization_options = p["customization_options"]
        option_to_image = dict()
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_name = option_name.lower()
                option_values = []
                for option_content in option_contents:
                    option_value = option_content["value"].strip().replace("/", " | ").lower()
                    option_image = option_content.get("image", None)
                    option_values.append(option_value)
                    option_to_image[option_value] = option_image
                options[option_name] = option_values
        products[i]["options"] = options
        products[i]["option_to_image"] = option_to_image

        if asin in attributes and "attributes" in attributes[asin]:
            products[i]["Attributes"] = attributes[asin]["attributes"]
        else:
            products[i]["Attributes"] = ["DUMMY_ATTR"]

        if human_goals:
            if asin in human_attributes:
                products[i]["instructions"] = human_attributes[asin]
        else:
            products[i]["instruction_text"] = attributes[asin].get("instruction", None)
            products[i]["instruction_attributes"] = attributes[asin].get("instruction_attributes", None)

        products[i]["MainImage"] = p["images"][0]
        products[i]["query"] = p["query"].lower().strip()
        all_products.append(products[i])

    for p in all_products:
        for a in p["Attributes"]:
            attribute_to_asins[a].add(p["asin"])

    _LAST_ALL_PRODUCTS = all_products
    product_item_dict = {p["asin"]: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    return all_products, product_item_dict, product_prices, attribute_to_asins
