# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Copyright 2025 verl-agent (GiGPO) Team
# Copyright 2026 AgentOCR Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SEARCH_TEMPLATE_NO_HIS = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
You should first conduct a reasoning process. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if you have sufficient information to answer the question with high confidence, provide your final answer within <answer> </answer> tags.
"""

SEARCH_TEMPLATE = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history, where <search>...</search> wrapped your past search queries and <information>...</information> wrapped the corresponding search results. History:
{memory_context}

Now it's your turn to respond for the current step.
You should first conduct a reasoning process. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if you have sufficient information to answer the question with high confidence, provide your final answer within <answer> </answer> tags.
"""


#######################################################################################




SEARCH_TEMPLATE_NO_HIS_OCR = """<image>
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

There is no prior search history yet. The image shows a compact task card for the current question rather than retrieved evidence.

Now it's your turn to respond for the current step.
You should first conduct a reasoning process. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if the question itself already provides sufficient information to answer with high confidence, provide your final answer within <answer> </answer> tags.
"""

SEARCH_TEMPLATE_NO_HIS_OCR_FORCE_SEARCH = """<image>
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

There is no prior search history yet. The image shows a compact task card for the current question rather than retrieved evidence.
Because this first-step image contains only the question and no supporting evidence, you MUST use this step to issue a search query.
Do not provide a final answer directly from the task card, even if the question looks familiar.
Do not guess hidden entities, dates, or titles from partial clues. If an intermediate entity is uncertain, search for it instead of inventing it.

Now it's your turn to respond for the current step.
You should first conduct a reasoning process, then call a search engine to get the missing evidence using format: <search> your query </search>.
This step must contain exactly one <search>...</search> action and no <answer>...</answer> tag anywhere.
"""

SEARCH_TEMPLATE_NO_HIS_OCR_STRICT_ENTITY_GROUNDING = """
First-search grounding:
- Keep the clearest anchor entity string exactly as written whenever possible.
- Write a concise retrieval query, not the full natural-language question.
- Keep year, role, location, title type, co-mentioned entity, or scope words when they are needed to disambiguate the target.
- For bridge or same-X questions, keep both anchors until the bridge entity is resolved.
- For age or current-age questions, do not invent a reference year or use current year/date placeholders.
- For institution member/composition questions, keep the full institution name together with member/composition wording.
"""

SEARCH_TEMPLATE_OCR = """<image>
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). 
The image contains the full history:
- Past queries are inside <search>...</search>
- Past results are inside <information>...</information>
- Past <search>...</search> text is only a retrieval attempt, not supporting evidence by itself.

Now it's your turn to respond for the current step.
You should first conduct a reasoning process. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if the image/history already provides sufficient, reliable information inside the retrieved <information> content to answer with high confidence, provide your final answer within <answer> </answer> tags.
Do not provide a final answer from a past query alone, from a vague query plan, or from world knowledge that is not explicitly supported by the shown history.
If an intermediate entity, title, person, or date is uncertain, search for it instead of guessing.
If the retrieved results are broad, off-target, or missing the final attribute, your next search should be a narrower follow-up query built from the exact entity string already shown in <information> plus the missing attribute you still need.
Do not keep repeating the full natural-language question if the current evidence already reveals a better anchor entity for the next search, but also do not over-compress away question constraints that are still needed to isolate the target.
Your final output must contain exactly one action tag:
- either one <search>...</search> and no <answer>...</answer>
- or one <answer>...</answer> and no <search>...</search>
Never output both tags in the same response.
"""

SEARCH_TEMPLATE_OCR_STRICT_ENTITY_GROUNDING = """
Grounding rules:
- Inside <answer>...</answer>, output only the minimal final answer span supported by the shown <information>.
- Copy entity, title, person, organization, location, work, or year strings exactly from the retrieved evidence.
- If the evidence is a near match, malformed, conflicting, or missing the exact asked attribute, issue another <search> instead of answering.
- For bridge questions, do not answer with the bridge entity itself; search again with the resolved entity plus the missing target attribute.
- For dynamic numeric questions such as age or years since, do not compute from an assumed current date; only answer an explicit value from evidence.
- For follow-up searches, keep exact anchor strings from the question or trusted evidence and preserve the disambiguating constraints still needed to isolate the target.
"""

SEARCH_COMPRESSION_TEMPLATE_NO_HIS = """
Additionally, select an image compression factor larger than or equal to 1.0 for the next image. Higher compression lowers cost, but too much compression harms image quality. If uncertain, output 1.0 to preserve readability. You must output the selected value within <compression> </compression> tags (e.g., <compression>1.0</compression>).
Output format:
1. Reasoning: state what the task card and question tell you.
2. <search>...</search> or <answer>...</answer>
3. <compression>...</compression>
"""

SEARCH_COMPRESSION_TEMPLATE_NO_HIS_FORCE_SEARCH = """
Additionally, select an image compression factor larger than or equal to 1.0 for the next image. Higher compression lowers cost, but too much compression harms image quality. If uncertain, output 1.0 to preserve readability. You must output the selected value within <compression> </compression> tags (e.g., <compression>1.0</compression>).
Output format:
1. Reasoning: state what the task card asks and what evidence you still need to retrieve.
2. Exactly one <search>...</search> tag. Do not output any <answer>...</answer> tag.
3. <compression>...</compression>
"""

SEARCH_COMPRESSION_TEMPLATE = """
Additionally, select an image compression factor larger than or equal to 1.0 for the next image. Higher compression lowers cost, but too much compression harms image quality. If uncertain, output 1.0 to preserve readability. You must output the selected value within <compression> </compression> tags (e.g., <compression>1.0</compression>).
Output format:
1. Reasoning: state what you found in the image.
2. Exactly one action tag: either one <search>...</search> or one <answer>...</answer>. Never output both.
3. <compression>...</compression>
"""
