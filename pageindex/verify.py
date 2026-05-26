"""Post-build leaf verification: does each leaf's title actually appear on its start_page?

If a leaf's title does NOT appear on its declared start_page, repair by searching
the pages between the nearest CORRECT leaves before and after it, and snap
start_page to the first page where the title is found. Each pass runs the string
check, then (when an LLM client is supplied) forwards any leaves it could not
confirm to an LLM verifier that knows about the common PDF extraction artifacts
("Incom e", scattered year columns in table headers). LLM approvals are sticky
across iterations, so a leaf, once confirmed, is treated as a bounding correct
neighbor and never moved. Combined with the string check (a leaf's text on its
own page does not depend on other leaves), this makes accuracy monotonic across
iterations; we stop as soon as a pass adds less than 1%.

Front-matter leaves (source "preface" / "toc", or node_id starting with "f")
carry conventional titles rather than text drawn from a page, so they auto-pass
the string check.
"""
import re
from typing import Dict, List, Optional, Set

from .builder import assign_end_pages
from .llm import LLMClient
from .pdf_utils import PDFDocument
from .prompts import VERIFY_TITLE_SYS, verify_title_user
from .tree import Node, iter_post_order


_MIN_NEEDLE = 5    # titles shorter than this would match almost any page
_NEEDLE_LEN = 40   # how much of the normalized title to use as the search needle


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _make_needle(title: str) -> Optional[str]:
    n = _normalize(title)[:_NEEDLE_LEN]
    return n if len(n) >= _MIN_NEEDLE else None


def _page_contains(pdf: PDFDocument, page: int, needle: str) -> bool:
    if page < 1 or page > pdf.page_count:
        return False
    return needle in _normalize(pdf.page_text(page))


def _ordered_leaves(root: Node) -> List[Node]:
    leaves: List[Node] = [n for n in iter_post_order(root) if n.is_leaf()]
    leaves.sort(key=lambda n: (n.start_page, n.end_page))
    return leaves


def _is_front_matter(node: Node) -> bool:
    """Front-matter leaves (preface, TOC) carry conventional titles, not text
    pulled from a page, so any string-vs-page check is meaningless for them.
    Detect via `source` (fresh builds) or the "f" id prefix (loaded trees).
    """
    return node.source in ("preface", "toc", "front_matter") \
        or node.node_id.startswith("f")


def verify_leaves(root: Node, pdf: PDFDocument,
                  confirmed_ids: Optional[Set[int]] = None) -> Dict:
    """Classify every leaf as correct / wrong / skipped.

    A leaf is `correct` if its title (normalized, first 40 chars) appears on
    `start_page`, `wrong` if not, and `skipped` if its normalized title is too
    short to be a reliable search needle. Skipped leaves are excluded from the
    accuracy denominator. Front-matter leaves auto-pass (conventional titles).

    `confirmed_ids` is a set of leaf `id()` values previously confirmed correct
    (string match, LLM, or front-matter rule). Those leaves are classified as
    `correct` without re-running the page-text check, so once a leaf is
    confirmed it is neither moved nor re-evaluated for the rest of the run.
    """
    confirmed_ids = confirmed_ids or set()
    leaves = _ordered_leaves(root)
    correct: List[Node] = []
    wrong: List[Node] = []
    skipped: List[Node] = []
    for leaf in leaves:
        if id(leaf) in confirmed_ids or _is_front_matter(leaf):
            correct.append(leaf)
            continue
        needle = _make_needle(leaf.title)
        if needle is None:
            skipped.append(leaf)
            continue
        if _page_contains(pdf, leaf.start_page, needle):
            correct.append(leaf)
        else:
            wrong.append(leaf)
    verifiable = len(correct) + len(wrong)
    accuracy = (len(correct) / verifiable) if verifiable else 1.0
    return {
        "leaves": leaves,
        "correct": correct,
        "wrong": wrong,
        "skipped": skipped,
        "verifiable": verifiable,
        "accuracy": accuracy,
    }


def repair_leaves(root: Node, pdf: PDFDocument, verification: Dict,
                  verbose: bool = True) -> int:
    """For every wrong leaf, sliding-window search between bounding correct
    leaves to find a page whose text contains the title. Mutates start_page
    in place. Returns the number of leaves whose start_page actually changed.
    """
    leaves = verification["leaves"]
    correct_ids = {id(n) for n in verification["correct"]}
    wrong_ids = {id(n) for n in verification["wrong"]}
    n = len(leaves)
    repairs = 0

    for i, leaf in enumerate(leaves):
        if id(leaf) not in wrong_ids:
            continue
        needle = _make_needle(leaf.title)
        if needle is None:
            continue

        lower = 1
        for j in range(i - 1, -1, -1):
            if id(leaves[j]) in correct_ids:
                lower = leaves[j].start_page + 1
                break

        upper = pdf.page_count
        for j in range(i + 1, n):
            if id(leaves[j]) in correct_ids:
                upper = max(lower, leaves[j].start_page - 1)
                break

        if lower > upper:
            continue

        for p in range(lower, upper + 1):
            if _page_contains(pdf, p, needle):
                if p != leaf.start_page:
                    if verbose:
                        preview = leaf.title[:60]
                        print(f"[repair] '{preview}' p.{leaf.start_page} -> p.{p}")
                    leaf.start_page = p
                    repairs += 1
                break

    if repairs:
        for node in iter_post_order(root):
            if node.children:
                node.children.sort(key=lambda c: c.start_page)
        assign_end_pages(root.children, pdf.page_count)
    return repairs


_MIN_GAIN_PER_ITER = 0.01           # stop when an iteration adds less than this
_LLM_VERIFY_MAX_PAGE_CHARS = 8000   # cap page text sent to the LLM verifier


def _set_accurate_flags(root: Node, correct_ids: set, skipped_ids: set) -> None:
    """Stamp `accurate` on every node from the final verification result.

    Leaf: True if it was verified correct OR couldn't be verified (skipped).
          False only when the title was searchable and didn't match start_page.
    Parent: True iff every descendant is True.
    """
    for node in iter_post_order(root):
        if node.is_leaf():
            node.accurate = (id(node) in correct_ids) or (id(node) in skipped_ids)
        else:
            node.accurate = all(c.accurate for c in node.children)


def _llm_verify_wrong(wrong: List[Node], pdf: PDFDocument, llm: LLMClient,
                      verbose: bool = True) -> Set[int]:
    """Ask the LLM whether each remaining wrong leaf's title appears on its page.

    Returns the set of leaf ids the LLM approves. The LLM prompt explicitly
    teaches the model about the artifacts the string match cannot handle
    (whitespace splits inside words, scattered year columns in table headers).
    Network/parse errors leave the leaf classified as wrong.
    """
    approved: Set[int] = set()
    for leaf in wrong:
        text = pdf.page_text(leaf.start_page)
        if not text.strip():
            continue
        if len(text) > _LLM_VERIFY_MAX_PAGE_CHARS:
            text = text[:_LLM_VERIFY_MAX_PAGE_CHARS] + "\n...[truncated]..."
        try:
            result = llm.complete_json(
                VERIFY_TITLE_SYS,
                verify_title_user(leaf.title, leaf.start_page, text),
                max_tokens=400,
            )
        except Exception as err:  # noqa: BLE001 - keep verifying remaining leaves
            if verbose:
                print(f"[llm_verify] error on '{leaf.title[:50]}': {err}")
            continue
        match = ""
        reason = ""
        if isinstance(result, dict):
            match = str(result.get("match", "")).strip().lower()
            reason = str(result.get("reason", "")).strip()
        preview = leaf.title[:60]
        if match == "yes":
            approved.add(id(leaf))
            if verbose:
                print(f"[llm_verify] ACCEPT '{preview}' on p.{leaf.start_page}"
                      + (f" ({reason})" if reason else ""))
        elif verbose:
            print(f"[llm_verify] REJECT '{preview}' on p.{leaf.start_page}"
                  + (f" ({reason})" if reason else ""))
    return approved


def verify_and_repair(root: Node, pdf: PDFDocument,
                      llm: Optional[LLMClient] = None, max_iters: int = 10,
                      verbose: bool = True) -> Dict:
    """Verify (string + LLM), repair wrong leaves, re-verify, repeat.

    Each pass runs the string check; any leaves it cannot confirm are forwarded
    to the LLM verifier (when an `llm` is supplied). LLM approvals are
    *sticky* across iterations - once a leaf is approved it is treated as
    correct for the rest of the run, so `repair_leaves` honors it as a
    bounding "correct" neighbor and never moves it. Combined with the string
    check (correct stays correct, since a leaf's text on its own page does not
    depend on other leaves), this makes per-iteration accuracy monotonic.

    Stop conditions:
      - reached 100%
      - repair iteration changed no leaves
      - per-iteration gain < 1%
      - hit `max_iters`

    Status values: `success` (100%), `acceptable` (>=60%), `below_60` (<60%).
    """
    confirmed_ids: Set[int] = set()

    def _verify_with_llm() -> Dict:
        """Verify, but never re-check already-confirmed leaves.

        Skips the string check for any leaf in `confirmed_ids`, then LLM-checks
        whatever the string pass left as wrong. Newly-confirmed leaves (by
        string match, by the LLM, or by the front-matter rule) are added to
        `confirmed_ids`, so subsequent iterations skip them entirely.
        """
        v = verify_leaves(root, pdf, confirmed_ids=confirmed_ids)
        confirmed_ids.update(id(leaf) for leaf in v["correct"])
        if llm is not None and v["wrong"]:
            if verbose:
                print(f"[llm_verify] checking {len(v['wrong'])} unconfirmed leaves")
            newly = _llm_verify_wrong(v["wrong"], pdf, llm, verbose=verbose)
            still_wrong: List[Node] = []
            for leaf in v["wrong"]:
                if id(leaf) in newly:
                    v["correct"].append(leaf)
                    confirmed_ids.add(id(leaf))
                else:
                    still_wrong.append(leaf)
            v["wrong"] = still_wrong
            v["verifiable"] = len(v["correct"]) + len(v["wrong"])
            v["accuracy"] = (len(v["correct"]) / v["verifiable"]) if v["verifiable"] else 1.0
        return v

    v = _verify_with_llm()
    initial = v["accuracy"]
    history = [initial]
    iterations = 0

    while iterations < max_iters and v["accuracy"] < 1.0:
        prev_acc = v["accuracy"]
        repaired = repair_leaves(root, pdf, v, verbose=verbose)
        if repaired == 0:
            if verbose:
                print(f"[verify] iter {iterations + 1}: no leaves changed, stopping")
            break

        iterations += 1
        v = _verify_with_llm()
        history.append(v["accuracy"])
        gain = v["accuracy"] - prev_acc
        if verbose:
            print(f"[verify] iter {iterations}: {prev_acc:.0%} -> {v['accuracy']:.0%} "
                  f"({repaired} leaf moves, +{gain:.1%})")

        if v["accuracy"] >= 1.0:
            break
        if gain < _MIN_GAIN_PER_ITER:
            if verbose:
                print(f"[verify] gain {gain:+.1%} < {_MIN_GAIN_PER_ITER:.0%}, stopping")
            break

    correct_ids = {id(n) for n in v["correct"]}
    skipped_ids = {id(n) for n in v["skipped"]}
    _set_accurate_flags(root, correct_ids, skipped_ids)

    final_acc = v["accuracy"]
    if final_acc >= 1.0:
        status = "success"
    elif final_acc >= 0.60:
        status = "acceptable"
    else:
        status = "below_60"

    return {
        "initial_accuracy": initial,
        "final_accuracy": final_acc,
        "history": history,
        "iterations": iterations,
        "verifiable": v["verifiable"],
        "status": status,
    }
