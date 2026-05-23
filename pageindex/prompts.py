"""All LLM prompt text lives here, kept apart from program logic."""
from .tree import Node

# --------------------------------------------------------------------------
# Sliding-window heading detection
# --------------------------------------------------------------------------
HEADING_SYS = (
    "You analyze a slice of a PDF and identify the section headings that begin "
    "within it. You are precise and never invent headings that are not present."
)


def heading_user(text: str, start: int, end: int) -> str:
    return f"""The text below is pages {start}-{end} of a PDF. Lines like `[page N]` mark page boundaries.

Identify every chapter / section / sub-section heading that STARTS within these pages.
For each heading give:
- "title": the heading text, cleaned of stray whitespace
- "level": hierarchy depth. 1 = top-level chapter or part, 2 = section, 3 = subsection, and so on. Be consistent across the whole document.
- "page": the page number it appears on (read it from the nearest preceding `[page N]` marker)

Ignore running headers/footers, page numbers, and figure/table captions.
If there are no headings in this slice, return an empty list.

Return JSON: {{"headings": [{{"title": "...", "level": 1, "page": {start}}}]}}

TEXT:
{text}
"""


# --------------------------------------------------------------------------
# Node summarization (bottom-up)
# --------------------------------------------------------------------------
SUMMARY_SYS = (
    "You write concise, information-dense summaries of document sections. "
    "Summaries are later used to route search queries, so they must be specific "
    "about what facts and topics a section contains."
)


def _title_clause(needs_title: bool) -> str:
    if not needs_title:
        return '"title" may be omitted.'
    return (
        'Also produce "title": a short descriptive title (at most 12 words) for this section.'
    )


def leaf_summary_user(node: Node, text: str, needs_title: bool) -> str:
    return f"""Section title: {node.title}
Pages: {node.start_page}-{node.end_page}

Write "summary": 3-6 sentences capturing the key topics, facts, and purpose of this
section. Be concrete - mention the specific things a reader could learn here.
{_title_clause(needs_title)}

Return JSON: {{"summary": "...", "title": "..."}}

SECTION TEXT:
{text}
"""


def parent_summary_user(node: Node, children_block: str, needs_title: bool) -> str:
    return f"""Section title: {node.title}
This section is composed of the following subsections (title: summary):

{children_block}

Write "summary": 3-6 sentences describing what this section as a whole covers,
based on its subsections. Be specific so it can be matched against search queries.
{_title_clause(needs_title)}

Return JSON: {{"summary": "...", "title": "..."}}
"""


# --------------------------------------------------------------------------
# Retrieval routing (descend the most relevant branch)
# --------------------------------------------------------------------------
ROUTING_SYS = (
    "You navigate a document tree to find the section most likely to answer a "
    "query. You pick exactly one branch to follow, or stop when no child is a "
    "better match than the current node."
)


def routing_user(query: str, node: Node, children_block: str) -> str:
    current = f'You are currently at node: "{node.title}"'
    if node.summary:
        current += f"\nIts summary: {node.summary}"
    return f"""Query: {query}

{current}

Candidate child branches:
{children_block}

Decide which single child branch is most likely to contain the answer.
- Choose "descend" and the child's index if one branch is the best path.
- Choose "stop" if none of the children are relevant, or if the current node
  itself already best matches the query (do not descend further).

Return JSON: {{"action": "descend" | "stop", "child_index": <int or null>, "reasoning": "..."}}
"""


# --------------------------------------------------------------------------
# Final answer generation
# --------------------------------------------------------------------------
ANSWER_SYS = (
    "You answer questions using only the supplied document excerpt. You cite "
    "page numbers and, if the excerpt does not contain the answer, you say so "
    "plainly instead of guessing."
)


def answer_user(query: str, node: Node, breadcrumb: str, context: str) -> str:
    return f"""Question: {query}

The retrieval system walked the document tree and selected this section:
  Path: {breadcrumb}
  Section: "{node.title}" (pages {node.start_page}-{node.end_page})

Answer the question using ONLY the excerpt below. Cite page numbers from the
`[page N]` markers where relevant. If the answer is not present in the excerpt,
say that this section does not contain it.

EXCERPT:
{context}
"""
