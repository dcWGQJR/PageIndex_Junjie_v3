"""LLM prompt text and user-prompt builders for the online evaluation pipeline.

Kept separate from `online.py` so prompt edits can be diffed cleanly and the
runner code stays focused on pipeline structure rather than long string
literals. Each agent (selection / answer / verifier / judge) is one
`*_SYS` constant plus one user-prompt-builder function.
"""
from typing import List

from pageindex.tree import Node


# --------------------------------------------------------------------------
# Planner agent: decompose the question into formula + needed sections
# before the selection agent sees the tree. The plan is appended to the
# selection user prompt as a hint, not a constraint.
# --------------------------------------------------------------------------
PLANNER_SYS = (
    "You decompose a financial question into the document sections needed "
    "to answer it. Output a short plan with: the kind of question "
    "(direct lookup vs derived metric); for derived metrics, the formula "
    "in line-item form (e.g. 'Inventory Turnover = COGS / Average "
    "Inventory'); the financial statements or notes whose line items are "
    "needed (e.g. Balance Sheet for Inventory, Income Statement for "
    "COGS); and the specific line items to look for. "
    "Be conservative: if you are not sure of the standard formula, leave "
    "it empty rather than guess - the selection agent will fall back to "
    "its own judgement. Use conventional section names (Balance Sheet, "
    "Income Statement, Cash Flow Statement, Statement of Equity, Notes "
    "to Financial Statements). "
    "For a direct lookup (e.g. 'What was 2022 revenue?'), set "
    "kind='direct_lookup', leave the formula empty, and list just the "
    "section and line item."
)


def planner_user(query: str) -> str:
    return f"""Question: {query}

Return JSON:
{{
  "kind": "direct_lookup" or "derived",
  "formula": "<line-item-level formula if derived, else empty string>",
  "needed_sections": ["<section name>", ...],
  "needed_line_items": ["<line item>", ...]
}}
"""


# --------------------------------------------------------------------------
# Selection agent: pick which tree nodes to feed into the answer prompt.
# --------------------------------------------------------------------------
SELECTION_SYS = (
    "You select the node(s) of a document tree most likely to contain the "
    "answer to a query based on both Summary and Title. If a node's title "
    "clearly names a term from the query but its summary describes a "
    "narrower or unrelated scope, still pick it - summaries can be "
    "incomplete. You are shown EVERY node in the tree at once - both "
    "parents (sections) and leaves (subsections) - with their titles, "
    "page ranges, and summaries. Pick whichever nodes are relevant; you "
    "may pick multiple when the answer plausibly draws on different "
    "sections (e.g. a financial statement AND a related note). "
    "Prefer the most specific node that still covers the answer: pick a "
    "child leaf rather than its parent when the leaf clearly contains the "
    "answer. Pick a parent only when the parent's own preamble is what's "
    "relevant. When the answer is spread across several of its children, "
    "pick the children one by one."
)


def selection_user(query: str, nodes_block: str, max_picks: int,
                   plan: str = "") -> str:
    plan_block = ""
    if plan:
        plan_block = (
            "A prior planning agent decomposed the question. Use it as "
            "guidance for which sections / line items to look for, but you "
            "may deviate if you see a clearly better fit in the tree.\n"
            f"{plan}\n\n"
        )
    return f"""Query: {query}

{plan_block}Every node in the document tree is listed below. Indentation reflects the
parent/child hierarchy. Each entry shows its index, title, page range, and
summary. Pick the nodes whose content is most likely to answer the query.

{nodes_block}

Return up to {max_picks} node indices, ordered by relevance (most relevant
first). Return fewer if only one or two are truly relevant - do not pad.
If NO node is clearly relevant, return an empty list.

Return JSON:
{{
  "node_indices": [<int>, ...],
  "reasoning": "<one short sentence>"
}}
"""


# --------------------------------------------------------------------------
# Answer agent: produce the final answer from the selected excerpts.
# --------------------------------------------------------------------------
ANSWER_MULTI_SYS = (
    "You answer financial questions. Prefer the provided document excerpts as "
    "the source of truth and cite page numbers when you use them. Multiple "
    "excerpts may be supplied from different sections of the same document - "
    "integrate them. If the excerpts do not fully cover the question, you may "
    "supplement with your own knowledge; clearly label any such content as "
    "outside the document. "
    "Pay particular attention to distinguishing similarly worded but different "
    "financial terms (e.g. gross vs. net, revenue vs. income, operating vs. "
    "free cash flow, total vs. long-term debt) - match the exact term in the "
    "question, do not substitute a near-synonym. "
    "Accounting convention: parentheses around a number indicate a negative "
    "value (e.g. (1,234) means -1,234)."
)


def excerpts_block(paths: List[List[Node]], excerpts: List[str]) -> str:
    """Numbered breadcrumb+excerpt block. Shared between the answer and
    verifier user prompts so both see identically-rendered evidence."""
    blocks = []
    for i, (path, excerpt) in enumerate(zip(paths, excerpts), start=1):
        breadcrumb = " > ".join(n.title for n in path)
        leaf = path[-1]
        blocks.append(
            f"### Excerpt {i}: {breadcrumb} (pages {leaf.start_page}-{leaf.end_page})\n"
            f"{excerpt}"
        )
    return "\n\n".join(blocks)


def answer_user_multi(query: str, paths: List[List[Node]], excerpts: List[str]) -> str:
    sections_block = excerpts_block(paths, excerpts)
    return f"""Question: {query}

The retrieval system selected {len(excerpts)} section(s) of the document.
Use the excerpts below as your primary source and cite page numbers from the
`[page N]` markers when drawing on them. If the excerpts do not cover the
question, you may answer from your own knowledge - just flag that part as
not coming from the document.

{sections_block}
"""


# --------------------------------------------------------------------------
# Judge agent: compare the predicted answer to the benchmark answer.
# --------------------------------------------------------------------------
JUDGE_SYS = (
    "You compare two answers to a financial question and decide whether they convey "
    "the same factual content. "
    "FIRST, extract the FINAL numeric or yes/no answer from the predicted reply - "
    "the prediction may include a long calculation walkthrough with many "
    "intermediate numbers; ignore those and compare only the final stated answer "
    "to the expected answer. "
    "Use your own financial-domain knowledge to judge contextually: recognize "
    "equivalent terminology (e.g. 'net sales' = 'revenue' in many filings, 'EBIT' "
    "= 'operating income'), standard formulas (e.g. gross margin = gross profit / "
    "revenue), and common units/derivations so a different phrasing of the same "
    "underlying fact is judged correct. "
    "Treat the following as equivalent: "
    "(a) unit/scale differences (e.g. $1,577M vs $1.58B, 1.2bn vs 1,200M); "
    "(b) fraction vs. percentage of the same ratio (e.g. 0.163 vs 16.3%, "
    "0.5 vs 50%); "
    "(c) a difference of +/-1 in the last reported digit at the same precision - "
    "ALWAYS treat this as equivalent, even if the question specifies a required decimal "
    "(d) rounding to different precisions when the question does NOT specify a "
    "required decimal accuracy - in that case, normalize both answers to the "
    "lower precision and treat them as equivalent if they match there, OR if "
    "they differ by +/- 1 in the last digit after rounding (e.g. with no precision specified, "
    "15.27% vs 15.3% vs 15% are all equivalent; 93.85 vs 93.9 vs 94 are all "
    "equivalent; $1,577M vs $1,580M vs $1.6B are all equivalent). Different "
    "decimal accuracies plus accumulated rounding from intermediate steps are "
    "expected when the question is open-ended on precision, and must not by "
    "themselves cause an F; "
    "Yes/No answers must match direction. If the predicted answer says the "
    "document does not contain the information but the expected answer is a "
    "concrete fact, the verdict is F."
)


def judge_prompt(question: str, expected: str, predicted: str) -> str:
    return f"""QUESTION:
{question}

EXPECTED ANSWER (ground truth):
{expected}

PREDICTED ANSWER:
{predicted}

Are these two answers substantively equivalent?
Respond with JSON: {{"verdict": "T" or "F", "reason": "<one short sentence>"}}
"""


# --------------------------------------------------------------------------
# Verifier agent: critique the predicted answer against the excerpts.
# --------------------------------------------------------------------------
VERIFY_ANSWER_SYS = (
    "You are a verifier for financial-question answers. You receive (a) the "
    "question, (b) the source excerpts the answerer was given, and (c) the "
    "predicted answer including its calculation walkthrough. "
    "Check three things AGAINST THE EXCERPTS ONLY, in order: "
    "  1. Line-item lookups: every number the prediction claims to read from "
    "     the document (e.g. 'Net sales: $519,926 million') must actually "
    "     appear in the excerpts, on the cited page, under the named line "
    "     item. Mismatched line items count as an issue. "
    "  2. Derivation choice: when the prediction DERIVES a value (e.g. "
    "     operating income = gross margin - SG&A - R&D), check that the same "
    "     value isn't ALREADY printed as a single line item in the excerpts. "
    "     Reading a line item directly is always preferred to a derived "
    "     reconstruction. "
    "  3. Arithmetic: every computation in the walkthrough (sums, averages, "
    "     ratios, percentages) must be correct to within 1 in the last "
    "     reported digit. Re-do the math step by step. "
    "Be CONSERVATIVE: only flag issues you are sure about - a flagged "
    "correct answer wastes a re-run and may degrade the final answer. If "
    "you cannot pin down a problem to a specific line, do not flag it. "
    "Return JSON: "
    "{\"ok\": true|false, "
    " \"issues\": [\"<one-line description per problem>\", ...], "
    " \"corrected_final_answer\": \"<the corrected final number/string if "
    "you are highly confident, otherwise empty string>\"}"
)


def verify_answer_prompt(query: str, excerpts: str, predicted: str) -> str:
    return f"""Question: {query}

Source excerpts the answerer used:
{excerpts}

Predicted answer (including its calculation walkthrough):
{predicted}

Verify per the system instructions and return JSON.
"""