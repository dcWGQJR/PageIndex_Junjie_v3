"""All LLM prompt text lives here, kept apart from program logic."""
from .tree import Node

# --------------------------------------------------------------------------
# Sliding-window heading detection
# --------------------------------------------------------------------------
HEADING_SYS = (
    "You analyze a slice of a PDF and identify the section headings that begin "
    "within it. Lines that the extractor judged visually heading-like (bold "
    "or noticeably larger than body text) are prefixed with `# ` in the input "
    "- treat these as strong candidate headings. You are precise and never "
    "invent headings that are not present."
)


def heading_user(text: str, start: int, end: int) -> str:
    return f"""The text below is pages {start}-{end} of a PDF. Lines like `[page N]` mark page boundaries. Lines prefixed with `# ` were flagged as visually heading-like (bold or larger font) by the extractor - they are strong candidates but you may still reject them if context shows they are not real headings.

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
# Printed table-of-contents parsing
# --------------------------------------------------------------------------
PRINTED_TOC_SYS = (
    "You convert a printed Table-of-Contents page into a clean, machine-readable "
    "JSON outline. You read every heading entry and the page number it points to."
)


def printed_toc_user(text: str, total_pages: int) -> str:
    return f"""The text below is the printed Table of Contents from a {total_pages}-page PDF.

IMPORTANT: a single TOC entry may be laid out across several lines because the
identifier, the section title, and the page number are in separate columns and
end up on separate lines after extraction. You MUST fuse them back into one
entry. For example, this raw text:

    PART I
    ITEM 1
    Business
    4
    ITEM 1A
    Risk Factors
    10
    ITEM 1B
    Unresolved Staff Comments
    12

represents THREE entries (Part I is a purely organizational label and is
DROPPED, not emitted):
    {{"title": "Item 1. Business",               "level": 1, "page": 4}}
    {{"title": "Item 1A. Risk Factors",          "level": 1, "page": 10}}
    {{"title": "Item 1B. Unresolved Staff Comments", "level": 1, "page": 12}}

Rules for each entry:
- "title":
    * Always combine the identifier ("Item 1", "Item 1A", "Note 5") with the
      descriptive section name on the next non-empty line into a single string
      like "Item 1A. Risk Factors". NEVER return a bare identifier like
      "Item 1" without the section name.
    * Use Title Case ("Item 1. Business"), not ALL CAPS ("ITEM 1 BUSINESS").
- "level":
    * 1 = a main item in the document outline ("Item 1. Business",
          "Item 1A. Risk Factors", "Note 1. Significant Accounting Policies").
    * 2+ = sub-items shown as indented bullets directly UNDER an item (e.g.
          the sub-list under "Item 7. MD&A": Overview, Results of Operations,
          Performance by Business Segment, ...).
- "page": the page number printed next to the heading, as an integer.

DROP these lines completely - do NOT emit entries for them:
- Organizational part labels like "PART I", "PART II", "PART III", "PART IV"
  (they are just groupings; the items inside them are what matters).
- The words "Table of Contents", "Beginning Page", "Page", column labels,
  blank rows, the document title, the company name, dates.

If a candidate entry has no page number, skip it.
Keep entries in the order they appear in the table.

Return JSON: {{"entries": [{{"title": "Part I", "level": 1, "page": 4}}, ...]}}

TABLE OF CONTENTS:
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


def parent_summary_user(node: Node, children_block: str, needs_title: bool,
                        extra_text: str = "") -> str:
    if extra_text.strip():
        extra_block = (
            "\nThis section also has preamble text (content in its page range "
            "that is NOT inside any subsection - typically introductory paragraphs "
            "before the first subsection):\n\n"
            f"{extra_text.strip()}\n"
        )
        basis = "based on its subsections AND the preamble text above"
    else:
        extra_block = ""
        basis = "based on its subsections"
    return f"""Section title: {node.title}
This section is composed of the following subsections (title: summary):

{children_block}
{extra_block}
Write "summary": 3-6 sentences describing what this section as a whole covers,
{basis}. Be specific so it can be matched against search queries.
{_title_clause(needs_title)}

Return JSON: {{"summary": "...", "title": "..."}}
"""


# --------------------------------------------------------------------------
# Leaf title verification (final pass over string-match failures)
# --------------------------------------------------------------------------
VERIFY_TITLE_SYS = (
    "You decide whether a given section title actually appears on a specific "
    "page of a PDF. PDF text extraction introduces predictable artifacts; you "
    "tolerate them and only say 'no' when the page clearly is not the start of "
    "that section."
)


def verify_title_user(title: str, page: int, page_text: str) -> str:
    return f"""Section title (from the document outline):
"{title}"

Claimed start page: {page}

Decide whether page {page} is the start of the section named by the title above.
Answer "yes" if the title appears on this page (allowing for the artifacts
below), "no" otherwise.

PDF text-extraction artifacts to tolerate when judging the match:
- Extra whitespace inside words: letters separated by spaces or split across
  lines, e.g. "Income" extracted as "Incom e" or "I n c o m e".
- Trailing year/date lists inside titles like
    "...for the years ended December 31, 2018, 2017 and 2016"  or
    "...at December 31, 2018 and 2017"
  are typically rendered as scattered column headers in a table (each year on
  its own line: "2018", "2017", "2016"). They will NOT appear as one contiguous
  string. Match the descriptive part of the title (e.g. "Consolidated Statement
  of Income ... December 31") and accept the trailing years if those individual
  years appear anywhere on the page.
- Filler/connector words ("for the", "at", "and") may be missing on the page
  even when present in the title; ignore them.
- Running headers/footers ("Table of Contents", page numbers) are noise; ignore
  them.
- Casing differences are irrelevant.

Return JSON: {{"match": "yes" | "no", "reason": "<one short sentence>"}}

PAGE {page} TEXT:
{page_text}
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
