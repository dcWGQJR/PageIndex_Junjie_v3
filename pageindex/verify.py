"""Post-build leaf verification: does each leaf's title actually appear on its start_page?

If a leaf's title does NOT appear on its declared start_page, repair by searching
the pages between the nearest CORRECT leaves before and after it, and snap
start_page to the first page where the title is found. Repeat until the tree
stops improving or 100% is reached. Revert any iteration that drops accuracy.

Front-matter leaves (source "preface" / node_id "f*") carry conventional titles
("preface", "toc") rather than text drawn from a page, so they auto-pass the
string check. Leaves that string-fail after repair are forwarded to an LLM
verifier (when an LLM client is supplied) that knows about the common PDF
extraction artifacts ("Incom e", scattered year columns in table headers).
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


def verify_leaves(root: Node, pdf: PDFDocument) -> Dict:
    """Classify every leaf as correct / wrong / skipped.

    A leaf is `correct` if its title (normalized, first 40 chars) appears on
    `start_page`, `wrong` if not, and `skipped` if its normalized title is too
    short to be a reliable search needle. Skipped leaves are excluded from the
    accuracy denominator. Front-matter leaves auto-pass (conventional titles).
    """
    leaves = _ordered_leaves(root)
    correct: List[Node] = []
    wrong: List[Node] = []
    skipped: List[Node] = []
    for leaf in leaves:
        if _is_front_matter(leaf):
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


def _snapshot(leaves: List[Node]) -> Dict[int, int]:
    return {id(n): n.start_page for n in leaves}


def _restore(leaves: List[Node], snap: Dict[int, int], pdf: PDFDocument, root: Node) -> None:
    for n in leaves:
        sp = snap.get(id(n))
        if sp is not None:
            n.start_page = sp
    for node in iter_post_order(root):
        if node.children:
            node.children.sort(key=lambda c: c.start_page)
    assign_end_pages(root.children, pdf.page_count)


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


_DROP_TOLERANCE = 0.02     # drops up to this are "fluctuation" -> revert+retry once
_STALL_AFTER_ITERS = 5     # after this many iters, require >= _STALL_MIN_GAIN improvement
_STALL_MIN_GAIN = 0.05     # otherwise stop
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
    """Verify, then loop repair/re-verify until 100%, stall, or sustained drop.

    Stop conditions (in priority order):
      - reached 100%
      - repair iteration changed no leaves
      - drop > 2% vs previous iter           -> revert to best, stop
      - drop <= 2% twice in a row             -> revert to best, stop
      - completed >= 5 iters with < 5% total improvement -> revert to best, stop
      - hit `max_iters`

    On a single small drop (<= 2%), revert to the best snapshot and retry one
    more time before giving up. The tree is always left at the best state seen.

    When `llm` is supplied, any leaves still classified `wrong` after the repair
    loop are forwarded to an LLM verifier that tolerates PDF-extraction
    artifacts (split words, scattered year columns) the string match cannot.

    Status values: `success` (100%), `acceptable` (>=60%), `below_60` (<60%).
    """
    v = verify_leaves(root, pdf)
    initial = v["accuracy"]
    history = [initial]
    leaves = v["leaves"]
    iterations = 0
    best_acc = initial
    best_snap = _snapshot(leaves)

    if v["verifiable"] > 0 and initial < 1.0:
        prev_acc = initial
        pending_retry = False

        while iterations < max_iters:
            repaired = repair_leaves(root, pdf, v, verbose=verbose)
            if repaired == 0:
                if verbose:
                    print(f"[verify] iter {iterations + 1}: no leaves changed, stopping")
                break

            iterations += 1
            v = verify_leaves(root, pdf)
            acc = v["accuracy"]
            history.append(acc)
            if verbose:
                print(f"[verify] iter {iterations}: {prev_acc:.0%} -> {acc:.0%} "
                      f"(best={best_acc:.0%}, {repaired} leaf moves)")

            if acc >= 1.0:
                best_acc = acc
                best_snap = _snapshot(leaves)
                break

            # Progressive floor: iter N (1..5) must reach at least N * 10%.
            if iterations <= 5 and acc < iterations * 0.10:
                if verbose:
                    print(f"[verify] iter {iterations}: acc {acc:.0%} below "
                          f"{iterations * 10}% floor, stopping")
                break

            drop = prev_acc - acc

            # Stall check: not enough overall progress after enough tries.
            if (iterations >= _STALL_AFTER_ITERS
                    and (best_acc - initial) < _STALL_MIN_GAIN):
                if verbose:
                    print(f"[verify] stalled: total improvement "
                          f"{(best_acc - initial):.1%} < {_STALL_MIN_GAIN:.0%} "
                          f"after {iterations} iters, stopping")
                break

            if drop > _DROP_TOLERANCE:
                if verbose:
                    print(f"[verify] drop {drop:.1%} > {_DROP_TOLERANCE:.0%}, stopping")
                break

            if drop > 0:
                if pending_retry:
                    if verbose:
                        print(f"[verify] second drop after retry, stopping")
                    break
                pending_retry = True
                if verbose:
                    print(f"[verify] small drop ({drop:.1%}), reverting to best "
                          f"({best_acc:.0%}) for retry")
                _restore(leaves, best_snap, pdf, root)
                v = verify_leaves(root, pdf)
                prev_acc = best_acc
                continue

            # acc >= prev_acc — progress (or flat). Record best.
            pending_retry = False
            if acc > best_acc:
                best_acc = acc
                best_snap = _snapshot(leaves)
            prev_acc = acc

    # Always leave the tree at the best state we saw.
    _restore(leaves, best_snap, pdf, root)

    # Final classification — also drives the `accurate` flag on every node.
    v_final = verify_leaves(root, pdf)
    correct_ids = {id(n) for n in v_final["correct"]}
    skipped_ids = {id(n) for n in v_final["skipped"]}

    llm_approved: Set[int] = set()
    if llm is not None and v_final["wrong"]:
        if verbose:
            print(f"[llm_verify] checking {len(v_final['wrong'])} leaves the "
                  f"string match could not confirm")
        llm_approved = _llm_verify_wrong(v_final["wrong"], pdf, llm, verbose=verbose)
        correct_ids |= llm_approved

    _set_accurate_flags(root, correct_ids, skipped_ids)

    if v_final["verifiable"]:
        final_acc = (len(v_final["correct"]) + len(llm_approved)) / v_final["verifiable"]
    else:
        final_acc = 1.0
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
        "verifiable": v_final["verifiable"],
        "status": status,
    }
