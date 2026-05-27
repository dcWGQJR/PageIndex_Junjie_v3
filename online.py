"""Answer FinanceBench questions against pre-built PageIndex trees and grade them.

Uses MULTI-BRANCH retrieval: at every level the router may pick up to `beam_size`
relevant child branches (default 2). The final answer is generated from the
combined text of every selected leaf, so a question whose answer lives in (for
example) "Note 18. Business Segments" is not lost when the router also follows
"Consolidated Statement of Cash Flows" - both excerpts reach the answer LLM.

For each row of `financebench_subset_QA.xlsx`:
- Load the pre-built tree at `trees/<doc_name>.tree.json`.
- Walk the tree via beam search; collect selected leaves with their paths.
- Concatenate the selected sections' text into one excerpt block; answer.
- Use an LLM judge to compare the predicted answer to the benchmark answer.
- Append a row to an output xlsx with: <all original columns> + predicted_answer
  + retrieval_path + verdict (T/F/?/SKIP/ERROR).

Usage
-----
    python online.py
    python online.py --limit 5
    python online.py --beam-size 3 --out results_beam3.xlsx
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import Workbook, load_workbook

from pageindex import Config, LLMClient
from pageindex.tree import Node, load_tree


# --------------------------------------------------------------------------
# Multi-branch retrieval prompts
# --------------------------------------------------------------------------
MULTI_ROUTING_SYS = (
    "You navigate a document tree to find the section(s) most likely to answer a "
    "query. You may pick MULTIPLE children when the answer plausibly needs "
    "information from more than one branch (e.g. the query references both a "
    "financial statement AND a related note). If no children are relevant, or "
    "the current node already best matches the query, choose to stop."
)


def _routing_user_multi(query: str, node: Node, children_block: str, max_picks: int) -> str:
    current = f'You are currently at node: "{node.title}"'
    if node.summary:
        current += f"\nSummary: {node.summary}"
    return f"""Query: {query}

{current}

Candidate child branches:
{children_block}

Pick up to {max_picks} child branches that look most likely to contain the
answer (return fewer if only one or two are truly relevant - do not pad).
If NONE of the children are relevant, return an empty list and action="stop".

Return JSON:
{{
  "action": "descend" or "stop",
  "child_indices": [<int>, ...],
  "reasoning": "<one short sentence>"
}}
"""


ANSWER_MULTI_SYS = (
    "You answer financial questions using only the provided document excerpts. "
    "Multiple excerpts may be supplied from different sections of the same "
    "document - integrate them. Cite page numbers in your answer. If none of "
    "the excerpts contain the answer, say so explicitly rather than guessing. "
    "Accounting convention: parentheses around a number indicate a negative "
    "value (e.g. (1,234) means -1,234)."
)


def _answer_user_multi(query: str, paths: List[List[Node]], excerpts: List[str]) -> str:
    blocks = []
    for i, (path, excerpt) in enumerate(zip(paths, excerpts), start=1):
        breadcrumb = " > ".join(n.title for n in path)
        leaf = path[-1]
        blocks.append(
            f"### Excerpt {i}: {breadcrumb} (pages {leaf.start_page}-{leaf.end_page})\n"
            f"{excerpt}"
        )
    sections_block = "\n\n".join(blocks)
    return f"""Question: {query}

The retrieval system selected {len(excerpts)} section(s) of the document.
Answer using ONLY the excerpts below. Cite page numbers from the `[page N]`
markers. If the answer is not contained in any of the excerpts, say the
selected sections do not contain it.

{sections_block}
"""


JUDGE_SYS = (
    "You compare two answers to a financial question and decide whether they convey "
    "the same factual content. Numeric values with reasonable rounding or unit "
    "differences (e.g. $1,577M vs $1.58B) are equivalent. Yes/No answers must match "
    "direction. If the predicted answer says the document does not contain the "
    "information but the expected answer is a concrete fact, the verdict is F."
)


def _judge_prompt(question: str, expected: str, predicted: str) -> str:
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
# Retrieval
# --------------------------------------------------------------------------
def _choose_children(llm: LLMClient, query: str, node: Node,
                     max_picks: int) -> Tuple[str, List[int]]:
    """Ask the LLM which children to descend into (or to stop)."""
    block = "\n".join(
        f"[{i}] {c.title} (pages {c.start_page}-{c.end_page})\n    {c.summary}"
        for i, c in enumerate(node.children)
    )
    result = llm.complete_json(
        MULTI_ROUTING_SYS, _routing_user_multi(query, node, block, max_picks)
    )
    result = result if isinstance(result, dict) else {}
    action = str(result.get("action", "descend")).strip().lower()
    if action not in ("descend", "stop"):
        action = "descend"
    raw = result.get("child_indices") or []
    if not isinstance(raw, list):
        raw = []
    indices: List[int] = []
    for item in raw[:max_picks]:
        try:
            ix = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= ix < len(node.children) and ix not in indices:
            indices.append(ix)
    return action, indices


def retrieve_multi(root: Node, query: str, config: Config, llm: LLMClient,
                   beam_size: int, max_leaves: int,
                   verbose: bool = False) -> List[List[Node]]:
    """Beam-style descent. Returns a list of paths from root to each selected node."""
    frontier: List[List[Node]] = [[root]]
    selected: List[List[Node]] = []
    depth = 0

    while frontier and depth < config.max_depth and len(selected) < max_leaves:
        next_frontier: List[List[Node]] = []
        for path in frontier:
            node = path[-1]
            if not node.children:
                selected.append(path)
                continue
            action, indices = _choose_children(llm, query, node, max_picks=beam_size)
            if verbose:
                print(f"[retrieve] d={depth} '{node.title}': {action} children={indices}")
            if action == "stop" or not indices:
                selected.append(path)
                continue
            for idx in indices:
                next_frontier.append(path + [node.children[idx]])
        # Cap beam so the frontier doesn't explode.
        frontier = next_frontier[:beam_size]
        depth += 1

    # Any unfinished frontier paths count as selected too.
    for path in frontier:
        if len(selected) >= max_leaves:
            break
        selected.append(path)

    # Dedupe by final-node identity, keep order.
    seen = set()
    final: List[List[Node]] = []
    for path in selected:
        key = id(path[-1])
        if key in seen:
            continue
        seen.add(key)
        final.append(path)
    return final[:max_leaves]


def _path_excerpt(path: List[Node]) -> str:
    """Concatenate ancestor preamble text and the leaf's text along `path`.

    Parent nodes carry "preamble" text - paragraphs in the parent's page range
    that no child covers, typically the introduction that precedes the first
    subsection. The router selects a leaf, but that leaf's content often only
    makes sense with the framing the parent set up (scope, definitions,
    accounting basis...), so we prepend each ancestor's non-empty text. The
    leaf comes last. Page markers (`[page N]`) inside each block are preserved
    because both leaf.text and parent.text come from pdf.text_range.
    """
    pieces: List[str] = []
    for node in path[:-1]:
        if node.text and node.text.strip():
            pieces.append(f'[Preamble from "{node.title}"]\n{node.text}')
    leaf = path[-1]
    if leaf.text:
        pieces.append(leaf.text)
    return "\n\n".join(pieces)


def answer_multi(root: Node, query: str, config: Config,
                 llm: LLMClient, beam_size: int = 2, max_leaves: int = 100,
                 verbose: bool = False,
                 answer_llm: Optional[LLMClient] = None) -> Dict[str, Any]:
    paths = retrieve_multi(root, query, config, llm, beam_size, max_leaves, verbose)
    excerpts = [_path_excerpt(p) for p in paths]
    breadcrumb = "  |  ".join(" > ".join(n.title for n in path) for path in paths)
    answer = (answer_llm or llm).complete(
        ANSWER_MULTI_SYS,
        _answer_user_multi(query, paths, excerpts),
        max_tokens=config.answer_max_tokens,
    )
    return {
        "answer": answer.strip(),
        "breadcrumb": breadcrumb,
        "paths": paths,
    }


# --------------------------------------------------------------------------
# Main runner
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Answer FinanceBench questions against pre-built PageIndex trees."
    )
    parser.add_argument("--qa-file", default="FinanceBench/financebench_subset_QA.xlsx")
    parser.add_argument("--trees-dir", default="trees")
    parser.add_argument("--out", default="financebench_results.xlsx")
    parser.add_argument("--limit", type=int, help="Process at most N rows.")
    parser.add_argument("--beam-size", type=int, default=100,
                        help="Max children to follow at each tree level (default: 100).")
    parser.add_argument("--max-leaves", type=int, default=100,
                        help="Max sections to feed into the answer (default: 100).")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Persist the output xlsx every N rows.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every routing decision.")
    args = parser.parse_args()

    qa_path = Path(args.qa_file)
    out_path = Path(args.out)
    trees_dir = Path(args.trees_dir)

    if not qa_path.is_file():
        print(f"Error: QA file not found: {qa_path}", file=sys.stderr)
        return 1

    wb = load_workbook(qa_path, data_only=True)
    ws = wb.active
    header = [c.value for c in ws[1]]
    required = ["doc_name", "question", "answer"]
    missing = [c for c in required if c not in header]
    if missing:
        print(f"Error: required columns missing from QA file: {missing}", file=sys.stderr)
        return 1
    col = {c: header.index(c) for c in required}

    out_wb = Workbook()
    out_ws = out_wb.active
    out_ws.title = "results"
    out_ws.append(header + ["predicted_answer", "retrieval_path", "verdict"])

    config = Config()
    llm = LLMClient(config)
    # Final answer uses a stronger model than routing/judge.
    answer_model = "gpt-4o-2024-11-20" if config.provider == "openai" else "claude-opus-4-7"
    answer_llm = LLMClient(replace(config, model=answer_model))

    n_total = ws.max_row - 1
    n_t = n_f = n_unknown = n_skipped = n_errored = 0
    tree_cache: Dict[str, Node] = {}

    for r, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if args.limit and (r - 1) > args.limit:
            break

        row_data = list(row)
        doc_name = row[col["doc_name"]]
        question = row[col["question"]]
        expected = row[col["answer"]]
        expected_str = "" if expected is None else str(expected).strip()

        if not doc_name or not question or not expected_str:
            n_skipped += 1
            out_ws.append(row_data + ["", "", "SKIP"])
            continue

        tree_path = trees_dir / f"{doc_name}.tree.json"

        if not tree_path.exists():
            n_skipped += 1
            reason = f"no tree at {tree_path}"
            print(f"[{r-1}/{n_total}] SKIP  {doc_name}  ({reason})")
            out_ws.append(row_data + ["", "", "SKIP"])
            continue

        if doc_name not in tree_cache:
            tree_cache[doc_name] = load_tree(str(tree_path))
        root = tree_cache[doc_name]

        t0 = time.time()
        try:
            result = answer_multi(
                root, str(question), config, llm,
                beam_size=args.beam_size, max_leaves=args.max_leaves,
                verbose=args.verbose, answer_llm=answer_llm,
            )
            predicted = result["answer"]
            breadcrumb = result["breadcrumb"]
        except Exception as err:  # noqa: BLE001 - keep batch alive
            n_errored += 1
            msg = str(err)[:300]
            print(f"[{r-1}/{n_total}] ERR   {doc_name}: {msg}")
            out_ws.append(row_data + ["", "", "ERROR"])
            if (r - 1) % args.save_every == 0:
                out_wb.save(out_path)
            continue

        try:
            judged = llm.complete_json(
                JUDGE_SYS, _judge_prompt(str(question), expected_str, predicted)
            )
            verdict = str(judged.get("verdict", "")).strip().upper()
            if verdict not in ("T", "F"):
                verdict = "?"
        except Exception:  # noqa: BLE001 - record judge failures
            verdict = "?"

        if verdict == "T":
            n_t += 1
        elif verdict == "F":
            n_f += 1
        else:
            n_unknown += 1

        dt = time.time() - t0
        q_text = str(question)
        q_preview = (q_text[:60] + "...") if len(q_text) > 60 else q_text
        judged_so_far = n_t + n_f
        running = f"running {n_t}/{judged_so_far}" if judged_so_far else "running -"
        print(f"[{r-1}/{n_total}] {verdict}  {doc_name}  ({dt:.1f}s)  [{running}]  q={q_preview!r}")

        out_ws.append(row_data + [predicted, breadcrumb, verdict])

        if verdict == "F":
            out_wb.save(out_path)
            justification = row[header.index("justification")] if "justification" in header else ""
            evidence = row[header.index("evidence")] if "evidence" in header else ""
            print(f"\n{'=' * 60}")
            print("First F detected - stopping.")
            print(f"{'=' * 60}")
            for field, value in [
                ("question", question),
                ("answer", expected_str),
                ("justification", justification),
                ("evidence", evidence),
                ("predicted_answer", predicted),
                ("retrieval_path", breadcrumb),
                ("verdict", verdict),
            ]:
                print(f"\n--- {field} ---\n{value}")

        if (r - 1) % args.save_every == 0:
            out_wb.save(out_path)

    out_wb.save(out_path)

    total_judged = n_t + n_f
    print(f"\n{'=' * 60}")
    print(f"Done.  T={n_t}  F={n_f}  ?={n_unknown}  skipped={n_skipped}  errored={n_errored}")
    if total_judged:
        print(f"Accuracy (T / (T+F)) = {n_t}/{total_judged} = {n_t / total_judged:.1%}")
    print(f"Results saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
