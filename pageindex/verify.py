"""Post-build leaf verification: LLM-only check that each leaf's title
actually appears on its declared start_page.

Pipeline (exactly two LLM-verify+repair passes, optionally a third rebuild):

  Pass 1: LLM-verify every leaf against `start_page`. Repair wrong leaves
          by sliding the title needle through
          [prev_correct.start_page, next_correct.start_page] (bounds
          INCLUSIVE - overlap with the bounding correct neighbors so a
          wrong leaf can land on the same page as a correct neighbor when
          two sections begin on one page).

  Pass 2: LLM-verify again. For leaves still wrong, search within the
          leaf's PARENT node's [start_page, end_page] range (so a wrong
          leaf can move beyond its bounding correct siblings but never
          outside the section it belongs to). After the repair completes,
          every parent's children list is re-sorted by start_page so the
          document order reflects the new positions.

  Optional rebuild: if final accuracy is still below 60%, discard the
          existing tree structure and rebuild from scratch using the
          sliding-window heading detector (`build_from_windows`). The new
          tree is then LLM-verified once so the returned metrics reflect
          its accuracy.

Front-matter leaves (source "preface" / "toc", or node_id starting with "f")
carry conventional titles rather than text drawn from a page, so they
auto-pass verification.

Confirmed leaves are sticky across passes: once a leaf is in the confirmed
set (LLM-approved or front-matter-rule), it is treated as correct for the
rest of the run and never re-verified or moved.
"""
import re
from typing import Dict, List, Optional, Set

from .builder import assign_end_pages, build_from_windows
from .config import Config
from .llm import LLMClient
from .pdf_utils import PDFDocument
from .prompts import VERIFY_TITLE_SYS, verify_title_user
from .tree import Node, iter_post_order


_MIN_NEEDLE = 5    # titles shorter than this would match almost any page
_NEEDLE_LEN = 40   # how much of the normalized title to use as the repair-scan needle


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _make_needle(title: str) -> Optional[str]:
    n = _normalize(title)[:_NEEDLE_LEN]
    return n if len(n) >= _MIN_NEEDLE else None


def _page_contains(pdf: PDFDocument, page: int, needle: str) -> bool:
    """Cheap string scan used during repair to score candidate pages.
    Verification itself never uses this - it's LLM-only."""
    if page < 1 or page > pdf.page_count:
        return False
    return needle in _normalize(pdf.page_text(page))


def _ordered_leaves(root: Node) -> List[Node]:
    leaves: List[Node] = [n for n in iter_post_order(root) if n.is_leaf()]
    leaves.sort(key=lambda n: (n.start_page, n.end_page))
    return leaves


def _is_front_matter(node: Node) -> bool:
    return node.source in ("preface", "toc", "front_matter") \
        or node.node_id.startswith("f")


# --------------------------------------------------------------------------
# LLM verification (no string-match shortcut, no page-text truncation)
# --------------------------------------------------------------------------
def _llm_verify_one(leaf: Node, pdf: PDFDocument, llm: LLMClient,
                    verbose: bool) -> bool:
    """Ask the LLM whether `leaf.title` is the start of a section on
    `leaf.start_page`. The full page text is sent (no truncation)."""
    text = pdf.page_text(leaf.start_page)
    if not text.strip():
        return False
    try:
        result = llm.complete_json(
            VERIFY_TITLE_SYS,
            verify_title_user(leaf.title, leaf.start_page, text),
            max_tokens=400,
        )
    except Exception as err:  # noqa: BLE001 - keep verifying remaining leaves
        if verbose:
            print(f"[llm_verify] error on '{leaf.title[:50]}': {err}")
        return False
    if not isinstance(result, dict):
        return False
    match = str(result.get("match", "")).strip().lower()
    reason = str(result.get("reason", "")).strip()
    if verbose:
        preview = leaf.title[:60]
        tag = "ACCEPT" if match == "yes" else "REJECT"
        suffix = f" ({reason})" if reason else ""
        print(f"[llm_verify] {tag} '{preview}' on p.{leaf.start_page}{suffix}")
    return match == "yes"


def verify_leaves(root: Node, pdf: PDFDocument, llm: LLMClient,
                  confirmed_ids: Optional[Set[int]] = None,
                  verbose: bool = True) -> Dict:
    """Classify every leaf as correct / wrong / skipped using the LLM only.

    A leaf is `correct` if the LLM confirms its title on `start_page`,
    `wrong` if not, and `skipped` if its normalized title is too short to
    make a meaningful repair-scan needle (`< 5` chars). Skipped leaves are
    excluded from the accuracy denominator.

    `confirmed_ids` contains leaves previously confirmed in earlier passes
    of this run. They are auto-classified as correct without an LLM call.
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
        if _make_needle(leaf.title) is None:
            skipped.append(leaf)
            continue
        if _llm_verify_one(leaf, pdf, llm, verbose=verbose):
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


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------
def _build_parent_map(root: Node) -> Dict[int, Node]:
    """Map id(child) -> its parent Node. The root itself is not in the map."""
    parent_map: Dict[int, Node] = {}
    for node in iter_post_order(root):
        for c in node.children:
            parent_map[id(c)] = node
    return parent_map


def repair_leaves(root: Node, pdf: PDFDocument, verification: Dict,
                  scope: str = "bounded", verbose: bool = True) -> int:
    """For every wrong leaf, scan a page range for a page containing the
    title needle, and snap `start_page` to it.

    `scope`:
      "bounded" (Pass 1): [prev_correct.start_page, next_correct.start_page],
        bounds INCLUSIVE so a wrong leaf can be placed on the same page as
        its bounding correct neighbor when two sections share a page.
      "parent"  (Pass 2): [leaf.parent.start_page, leaf.parent.end_page] -
        the leaf can move beyond its bounding correct siblings but never
        outside the section it belongs to. Leaves directly under root use
        [1, pdf.page_count] (i.e. the whole document) since the root spans
        the entire PDF.

    After the pass, parents' children are re-sorted by start_page and
    `assign_end_pages` is rerun so the structural ranges stay consistent
    with the new positions.
    """
    leaves = verification["leaves"]
    correct_ids = {id(n) for n in verification["correct"]}
    wrong_ids = {id(n) for n in verification["wrong"]}
    n = len(leaves)
    repairs = 0
    parent_map = _build_parent_map(root) if scope == "parent" else {}

    for i, leaf in enumerate(leaves):
        if id(leaf) not in wrong_ids:
            continue
        needle = _make_needle(leaf.title)
        if needle is None:
            continue

        if scope == "parent":
            parent = parent_map.get(id(leaf))
            if parent is None:
                # Leaf is a direct child of root - root spans the whole PDF.
                lower, upper = 1, pdf.page_count
            else:
                lower = max(1, parent.start_page)
                upper = min(pdf.page_count, parent.end_page)
                if lower > upper:
                    continue
        else:
            lower = 1
            for j in range(i - 1, -1, -1):
                if id(leaves[j]) in correct_ids:
                    lower = leaves[j].start_page
                    break
            upper = pdf.page_count
            for j in range(i + 1, n):
                if id(leaves[j]) in correct_ids:
                    upper = max(lower, leaves[j].start_page)
                    break
            if lower > upper:
                continue

        for p in range(lower, upper + 1):
            if _page_contains(pdf, p, needle):
                if p != leaf.start_page:
                    if verbose:
                        preview = leaf.title[:60]
                        print(f"[repair {scope}] '{preview}' "
                              f"p.{leaf.start_page} -> p.{p}")
                    leaf.start_page = p
                    repairs += 1
                break

    if repairs:
        for node in iter_post_order(root):
            if node.children:
                node.children.sort(key=lambda c: c.start_page)
        assign_end_pages(root.children, pdf.page_count)
    return repairs


# --------------------------------------------------------------------------
# refine_end_pages and _set_accurate_flags (unchanged from prior version)
# --------------------------------------------------------------------------
_MIDPAGE_OFFSET_CHARS = 200


def _starts_at_top_of_page(pdf: PDFDocument, page: int, title: str) -> bool:
    needle = _make_needle(title)
    if needle is None:
        return True
    text = pdf.page_text(page)
    if not text:
        return True
    pos = _normalize(text).find(needle)
    if pos == -1:
        return True
    return pos < _MIDPAGE_OFFSET_CHARS


def refine_end_pages(nodes: List[Node], parent_end: int,
                     pdf: PDFDocument) -> int:
    """Re-derive end_page accounting for mid-page section starts.

    `assign_end_pages` ends a node one page before its next sibling. When
    the next sibling's title appears mid-page, the trailing portion of that
    page belongs to the current sibling, so we extend its end_page by one.
    """
    nodes = sorted(nodes, key=lambda c: c.start_page)
    changes = 0
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            nxt = nodes[i + 1]
            if _starts_at_top_of_page(pdf, nxt.start_page, nxt.title):
                new_end = max(node.start_page, nxt.start_page - 1)
            else:
                new_end = max(node.start_page, nxt.start_page)
        else:
            new_end = max(node.start_page, parent_end)
        if new_end != node.end_page:
            node.end_page = new_end
            changes += 1
        if node.children:
            changes += refine_end_pages(node.children, node.end_page, pdf)
    return changes


def _set_accurate_flags(root: Node, correct_ids: set, skipped_ids: set) -> None:
    for node in iter_post_order(root):
        if node.is_leaf():
            node.accurate = (id(node) in correct_ids) or (id(node) in skipped_ids)
        else:
            node.accurate = all(c.accurate for c in node.children)


# --------------------------------------------------------------------------
# Orchestration: pass 1 (bounded) -> pass 2 (whole-PDF) -> optional rebuild
# --------------------------------------------------------------------------
_REBUILD_THRESHOLD = 0.60


def verify_and_repair(root: Node, pdf: PDFDocument, config: Config,
                      llm: LLMClient, verbose: bool = True) -> Dict:
    """Two LLM-verify+repair passes, with an optional sliding-window rebuild
    when the final accuracy is below `_REBUILD_THRESHOLD`.

    Returns metrics: initial_accuracy, final_accuracy, history (list of
    accuracies after each verify), iterations (1, 2, or 3), verifiable
    (final leaf count), status ("success" / "acceptable" / "below_60"),
    rebuilt (True if pass 3 fired).
    """
    if llm is None:
        raise ValueError("verify_and_repair now requires an LLM client "
                         "(string-match verification has been removed).")

    confirmed_ids: Set[int] = set()

    def _verify() -> Dict:
        v = verify_leaves(root, pdf, llm,
                          confirmed_ids=confirmed_ids, verbose=verbose)
        confirmed_ids.update(id(leaf) for leaf in v["correct"])
        return v

    # --- Pass 1: LLM verify + bounded repair ----------------------------
    v = _verify()
    initial = v["accuracy"]
    history: List[float] = [initial]
    if verbose:
        print(f"[verify] pass 1 (bounded): {len(v['correct'])}/{v['verifiable']} "
              f"correct ({v['accuracy']:.0%})")

    if v["accuracy"] < 1.0:
        repaired = repair_leaves(root, pdf, v, scope="bounded", verbose=verbose)
        if verbose:
            print(f"[verify] pass 1 repair: {repaired} leaf move(s)")

    # --- Pass 2: LLM verify + parent-bounded repair ---------------------
    v = _verify()
    history.append(v["accuracy"])
    if verbose:
        print(f"[verify] pass 2 (parent): {len(v['correct'])}/{v['verifiable']} "
              f"correct ({v['accuracy']:.0%})")

    if v["accuracy"] < 1.0:
        repaired = repair_leaves(root, pdf, v, scope="parent", verbose=verbose)
        if verbose:
            print(f"[verify] pass 2 repair: {repaired} leaf move(s) (parent-bounded)")
        # Re-verify after the parent-scoped moves so the metrics reflect
        # the post-repair state and the rebuild decision below uses it.
        v = _verify()
        history.append(v["accuracy"])

    iterations = 2
    rebuilt = False

    # --- Optional Pass 3: rebuild via sliding-window heading detection --
    if v["accuracy"] < _REBUILD_THRESHOLD:
        if verbose:
            print(f"[verify] accuracy {v['accuracy']:.0%} < "
                  f"{_REBUILD_THRESHOLD:.0%} - rebuilding tree via sliding window")
        new_children = build_from_windows(pdf, config, llm, verbose=verbose)
        root.children = new_children
        assign_end_pages(root.children, pdf.page_count)
        # Reset confirmation set: the old node identities are gone.
        confirmed_ids.clear()
        v = _verify()
        history.append(v["accuracy"])
        iterations = 3
        rebuilt = True
        if verbose:
            print(f"[verify] post-rebuild: {len(v['correct'])}/{v['verifiable']} "
                  f"correct ({v['accuracy']:.0%})")

    # --- End-page refinement + accurate flags ---------------------------
    refined = refine_end_pages(root.children, pdf.page_count, pdf)
    if verbose and refined:
        print(f"[refine] bumped end_page on {refined} node(s) for mid-page next starts")

    correct_ids = {id(n) for n in v["correct"]}
    skipped_ids = {id(n) for n in v["skipped"]}
    _set_accurate_flags(root, correct_ids, skipped_ids)

    final_acc = v["accuracy"]
    if final_acc >= 1.0:
        status = "success"
    elif final_acc >= _REBUILD_THRESHOLD:
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
        "rebuilt": rebuilt,
    }
