"""Tree construction: from the PDF's embedded TOC, the printed TOC, or sliding window."""
import math
import os
import re
from typing import Dict, List, Optional, Tuple

from .config import Config
from .llm import LLMClient
from .pdf_utils import PDFDocument
from .prompts import (
    HEADING_SYS,
    PRINTED_TOC_SYS,
    SUMMARY_SYS,
    heading_user,
    leaf_summary_user,
    parent_summary_user,
    printed_toc_user,
)
from .tree import Node, iter_post_order


# --------------------------------------------------------------------------
# Shared structural helpers
# --------------------------------------------------------------------------
def build_hierarchy(headings: List[Dict], source: str, id_prefix: str) -> List[Node]:
    """Turn a flat, in-document-order list of headings into a nested node list.

    Each heading is `{level, title, page}`. A heading nests under the most
    recent open heading whose level is strictly smaller.
    """
    root_children: List[Node] = []
    stack: List[Node] = []
    for i, h in enumerate(headings, start=1):
        node = Node(
            node_id=f"{id_prefix}{i}",
            title=h["title"],
            level=h["level"],
            start_page=h["page"],
            end_page=h["page"],
            source=source,
        )
        while stack and stack[-1].level >= node.level:
            stack.pop()
        (stack[-1].add_child(node) if stack else root_children.append(node))
        stack.append(node)
    return root_children


def assign_end_pages(nodes: List[Node], parent_end: int) -> None:
    """Fill in `end_page` for a sibling list and recurse into children.

    A node ends just before its next sibling starts; the last sibling inherits
    the parent's end. This makes every node's range span all its descendants.
    """
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            node.end_page = max(node.start_page, nodes[i + 1].start_page - 1)
        else:
            node.end_page = max(node.start_page, parent_end)
        if node.children:
            assign_end_pages(node.children, node.end_page)


def build_front_matter(pdf: PDFDocument, start: int, end: int) -> List[Node]:
    """Window the pages before the first content heading into preface / toc nodes.

    Pages before the first content heading are split into a "preface" node and
    (if a printed Table of Contents is detected within the range) a separate
    "toc" node, so the TOC is never absorbed into the preface. Conventional
    titles ("preface", "toc") are used; they are not derived from page text.
    """
    if start > end:
        return []

    toc_pages = [p for p in pdf.find_printed_toc_pages() if start <= p <= end]
    toc_lo = min(toc_pages) if toc_pages else None
    toc_hi = max(toc_pages) if toc_pages else None

    spans: List[tuple] = []  # (kind, start, end)
    if toc_lo is None:
        spans.append(("preface", start, end))
    else:
        if toc_lo > start:
            spans.append(("preface", start, toc_lo - 1))
        spans.append(("toc", toc_lo, toc_hi))
        if toc_hi < end:
            spans.append(("preface", toc_hi + 1, end))

    nodes: List[Node] = []
    for i, (kind, s, e) in enumerate(spans, start=1):
        nodes.append(Node(
            node_id=f"f{i}",
            title=kind,
            level=1,
            start_page=s,
            end_page=e,
            source=kind,
        ))
    return nodes


# --------------------------------------------------------------------------
# Path 1: build from the embedded table of contents (PDF bookmark outline)
# --------------------------------------------------------------------------
def build_from_toc(pdf: PDFDocument, toc: List[Dict]) -> List[Node]:
    headings = [
        {
            "level": max(1, e["level"]),
            "title": e["title"],
            "page": min(max(1, e["page"]), pdf.page_count),
        }
        for e in toc
    ]
    front = build_front_matter(pdf, 1, headings[0]["page"] - 1)
    content = build_hierarchy(headings, source="embedded_toc", id_prefix="t")
    return front + content


# --------------------------------------------------------------------------
# Path 2: build from a printed (in-body) table-of-contents page
# --------------------------------------------------------------------------
# A title that is *just* an item identifier ("Item 1", "Item 1A.") with no
# descriptive section name. Such entries mean the LLM failed to fuse the
# multi-line layout - we drop them rather than carry a half-entry forward.
_BARE_ITEM_TITLE = re.compile(r"^\s*item\s+\d+[a-z]?\.?\s*$", re.IGNORECASE)

# A title that is only an organizational part label ("Part I", "PART II.").
# These are groupings in SEC filings with no content of their own - we don't
# want them as nodes; the items inside them become the top-level nodes.
_PART_ONLY_TITLE = re.compile(r"^\s*part\s+([ivxlcdm]+|\d+)\.?\s*$", re.IGNORECASE)


def _detect_page_offset(pdf: PDFDocument, headings: List[Dict],
                        toc_pages: List[int], max_search: int = 60) -> int:
    """Estimate `(physical - printed)` page offset by locating the first heading.

    The page numbers a printed TOC lists are document-print page numbers, which
    often differ from PDF page indices (cover, TOC etc. shift everything by a
    few pages). We find where the first heading actually appears in the PDF and
    return the implied shift.
    """
    if not headings:
        return 0
    needle = re.sub(r"[^a-zA-Z0-9]", "", headings[0]["title"]).lower()[:40]
    if len(needle) < 5:
        return 0
    last_toc = max(toc_pages) if toc_pages else 0
    upper = min(last_toc + max_search, pdf.page_count)
    for p in range(last_toc + 1, upper + 1):
        text = re.sub(r"[^a-zA-Z0-9]", "", pdf.page_text(p)).lower()
        if needle in text:
            return p - headings[0]["page"]
    return 0


# If the parsed TOC's last entry sits earlier than this fraction of the PDF,
# look for an additional printed TOC block beyond the initial scan window.
_LATE_TOC_COVERAGE = 2 / 3


def _parse_printed_toc_block(pdf: PDFDocument, llm: LLMClient,
                             toc_pages: List[int],
                             verbose: bool) -> List[Dict]:
    """LLM-parse one contiguous printed TOC block and return clean headings.

    Drops bare-identifier and Part-only entries; clamps page numbers to the
    document range. Page-offset correction is applied separately by the
    caller using `_detect_page_offset`.
    """
    text = pdf.text_range(toc_pages[0], toc_pages[-1])
    result = llm.complete_json(PRINTED_TOC_SYS,
                               printed_toc_user(text, pdf.page_count))
    raw = result.get("entries", []) if isinstance(result, dict) else result

    headings: List[Dict] = []
    dropped_bare = 0
    dropped_parts = 0
    for e in raw or []:
        try:
            page = int(e["page"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (1 <= page <= pdf.page_count):
            continue
        title = str(e.get("title", "")).strip()
        if not title:
            continue
        if _BARE_ITEM_TITLE.match(title):
            dropped_bare += 1
            continue
        if _PART_ONLY_TITLE.match(title):
            dropped_parts += 1
            continue
        level = max(1, int(e.get("level", 1)))
        headings.append({"level": level, "title": title, "page": page})

    if verbose and dropped_bare:
        print(f"[build] dropped {dropped_bare} bare-identifier TOC entries "
              f"(e.g. 'Item 1' with no description)")
    if verbose and dropped_parts:
        print(f"[build] dropped {dropped_parts} organizational 'Part N' TOC entries")
    return headings


def build_from_printed_toc(pdf: PDFDocument, config: Config, llm: LLMClient,
                           verbose: bool = True) -> Optional[List[Node]]:
    """Scan early pages for a printed TOC, LLM-parse it, then build the hierarchy.

    If the parsed TOC's last entry sits earlier than ~2/3 of the document,
    scan beyond the initial window for an additional printed TOC block
    (common in long filings that have a mid-document TOC). Both blocks'
    headings are merged before the hierarchy is built.

    Returns the list of root-level Nodes, or None if no usable printed TOC
    could be extracted (caller falls back to the next mode).
    """
    toc_pages = pdf.find_printed_toc_pages()
    if not toc_pages:
        if verbose:
            print("[build] no printed TOC page detected")
        return None
    if verbose:
        print(f"[build] candidate printed TOC pages: {toc_pages}")

    headings = _parse_printed_toc_block(pdf, llm, toc_pages, verbose)

    if len(headings) < config.toc_min_entries:
        if verbose:
            print(f"[build] printed TOC yielded only {len(headings)} entries; skipping")
        return None

    offset = _detect_page_offset(pdf, headings, toc_pages)
    if offset:
        if verbose:
            print(f"[build] printed-to-physical page offset: {offset:+d}")
        for h in headings:
            h["page"] = min(pdf.page_count, max(1, h["page"] + offset))

    # If the first TOC's coverage stops well before the end of the document,
    # look for additional TOC pages beyond the initial scan window. This
    # catches filings that have a separate mid-document TOC (e.g. an attached
    # exhibit, a financial-statements TOC after the main 10-K body, etc.).
    coverage_cutoff = int(pdf.page_count * _LATE_TOC_COVERAGE)
    max_heading_page = max((h["page"] for h in headings), default=0)
    if max_heading_page < coverage_cutoff:
        scan_from = max(toc_pages) + 1
        remaining = pdf.page_count - scan_from + 1
        if remaining > 0:
            if verbose:
                print(f"[build] last TOC entry at p.{max_heading_page} < p.{coverage_cutoff} "
                      f"({_LATE_TOC_COVERAGE:.0%} of {pdf.page_count}); "
                      f"scanning p.{scan_from}-{pdf.page_count} for another TOC")
            extra_pages = pdf.find_printed_toc_pages(
                max_scan=remaining, start_page=scan_from,
            )
            if extra_pages:
                if verbose:
                    print(f"[build] additional printed TOC pages: {extra_pages}")
                extra_headings = _parse_printed_toc_block(pdf, llm, extra_pages, verbose)
                extra_offset = _detect_page_offset(pdf, extra_headings, extra_pages)
                if extra_offset:
                    if verbose:
                        print(f"[build] extra-TOC page offset: {extra_offset:+d}")
                    for h in extra_headings:
                        h["page"] = min(pdf.page_count, max(1, h["page"] + extra_offset))
                # Only keep extra headings whose corrected page falls AFTER
                # the first TOC's coverage - same-section dupes would just
                # produce overlapping levels.
                extra_headings = [h for h in extra_headings if h["page"] > max_heading_page]
                if extra_headings:
                    if verbose:
                        print(f"[build] merging {len(extra_headings)} late-TOC entries")
                    headings.extend(extra_headings)
            elif verbose:
                print(f"[build] no additional TOC found beyond p.{scan_from}")

    headings.sort(key=lambda h: (h["page"], h["level"]))
    if verbose:
        print(f"[build] using printed TOC ({len(headings)} entries)")

    front = build_front_matter(pdf, 1, headings[0]["page"] - 1)
    content = build_hierarchy(headings, source="printed_toc", id_prefix="p")
    return front + content


# --------------------------------------------------------------------------
# Path 2: build by sliding a window over the document
# --------------------------------------------------------------------------
def _detect_headings(llm: LLMClient, text: str, start: int, end: int) -> List[Dict]:
    result = llm.complete_json(HEADING_SYS, heading_user(text, start, end))
    items = result.get("headings", []) if isinstance(result, dict) else result
    out: List[Dict] = []
    for it in items or []:
        try:
            out.append({
                "title": str(it["title"]).strip(),
                "level": int(it.get("level", 1)),
                "page": int(it["page"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _chunk_nodes(pdf: PDFDocument, config: Config) -> List[Node]:
    """Fallback when no headings can be found: flat fixed-size page chunks."""
    nodes: List[Node] = []
    page, i = 1, 0
    while page <= pdf.page_count:
        w_end = min(page + config.window_size - 1, pdf.page_count)
        i += 1
        nodes.append(Node(
            node_id=f"w{i}",
            title=f"Pages {page}-{w_end}",
            level=1,
            start_page=page,
            end_page=w_end,
            source="window",
        ))
        page = w_end + 1
    return nodes


def build_from_windows(pdf: PDFDocument, config: Config, llm: LLMClient,
                       verbose: bool = True) -> List[Node]:
    headings: List[Dict] = []
    seen = set()
    page = 1
    while page <= pdf.page_count:
        w_end = min(page + config.window_size - 1, pdf.page_count)
        if verbose:
            print(f"[window] analyzing pages {page}-{w_end}")
        text = pdf.text_range(page, w_end, config.max_chars_per_block)
        for h in _detect_headings(llm, text, page, w_end):
            h["page"] = min(max(1, h["page"]), pdf.page_count)
            h["level"] = max(1, h["level"])
            key = (h["title"].lower(), h["page"])
            if not h["title"] or key in seen:
                continue
            seen.add(key)
            headings.append(h)
        if w_end >= pdf.page_count:
            break
        page = max(page + 1, w_end + 1 - config.window_overlap)

    if not headings:
        if verbose:
            print("[window] no headings detected - falling back to fixed page chunks")
        return _chunk_nodes(pdf, config)

    headings.sort(key=lambda h: (h["page"], h["level"]))
    front = build_front_matter(pdf, 1, headings[0]["page"] - 1)
    content = build_hierarchy(headings, source="window", id_prefix="w")
    return front + content


# --------------------------------------------------------------------------
# Orchestration: choose a path, then summarize bottom-up
# --------------------------------------------------------------------------
def build_tree(pdf: PDFDocument, config: Config, llm: LLMClient,
               mode: str = "auto", verbose: bool = True) -> Tuple[Node, str]:
    """Build the structural tree (no summaries yet). Returns (root, mode_used).

    Modes
    -----
    - "auto"     : try embedded outline, then printed TOC, then sliding-window.
    - "toc"      : try embedded outline, then printed TOC; error if neither works.
    - "embedded" : only the PDF's embedded outline (bookmarks).
    - "printed"  : only a printed (in-body) TOC page.
    - "window"   : skip TOC paths and use the sliding-window LLM scan.
    """
    if mode not in ("auto", "toc", "embedded", "printed", "window"):
        raise ValueError(f"unknown build mode {mode!r}")

    root = Node(
        node_id="root",
        title=os.path.basename(pdf.path),
        level=0,
        start_page=1,
        end_page=pdf.page_count,
        source="root",
    )

    children = None
    used: Optional[str] = None

    # 1. Embedded outline (cheap; no LLM call).
    if mode in ("auto", "toc", "embedded"):
        toc = pdf.embedded_toc()
        if len(toc) >= config.toc_min_entries:
            if verbose:
                print(f"[build] using embedded outline ({len(toc)} entries)")
            children = build_from_toc(pdf, toc)
            used = "embedded_toc"
        elif mode == "embedded":
            raise ValueError("PDF has no usable embedded outline.")

    # 2. Printed in-body TOC (one LLM call to parse it).
    if children is None and mode in ("auto", "toc", "printed"):
        children = build_from_printed_toc(pdf, config, llm, verbose=verbose)
        if children is not None:
            used = "printed_toc"
        elif mode in ("toc", "printed"):
            raise ValueError(
                "PDF has no usable table of contents "
                "(neither an embedded outline nor a parseable printed TOC)."
            )

    # 3. Sliding-window LLM scan over the whole document.
    if children is None:
        if verbose:
            print("[build] analyzing document with a sliding window")
        children = build_from_windows(pdf, config, llm, verbose=verbose)
        used = "window"

    root.children = children
    assign_end_pages(root.children, pdf.page_count)
    return root, used


# --------------------------------------------------------------------------
# Post-build: split oversized leaves with the sliding-window heading detector
# --------------------------------------------------------------------------
def _is_front_matter_node(node: Node) -> bool:
    return node.source in ("preface", "toc", "front_matter") or node.node_id.startswith("f")


def _normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _scan_subheadings(pdf: PDFDocument, config: Config, llm: LLMClient,
                      leaf: Node, verbose: bool) -> List[Dict]:
    """Slide a window over a leaf's page range and collect inner sub-headings.

    Returns headings whose page falls strictly INSIDE the leaf range (not on
    the leaf's own start_page, since that would just re-detect the leaf
    itself). Drops headings whose normalized title matches the leaf's title
    or another already-collected heading.
    """
    span = leaf.end_page - leaf.start_page + 1
    if span <= config.window_size:
        return []

    parent_norm = _normalize_title(leaf.title)
    collected: List[Dict] = []
    seen: set = set()
    page = leaf.start_page
    while page <= leaf.end_page:
        w_end = min(page + config.window_size - 1, leaf.end_page)
        try:
            text = pdf.text_range(page, w_end, config.max_chars_per_block)
            heads = _detect_headings(llm, text, page, w_end)
        except Exception as err:  # noqa: BLE001 - degrade to "no headings here"
            if verbose:
                print(f"[split] window p.{page}-{w_end} of '{leaf.title[:50]}': "
                      f"detect error: {err}")
            heads = []

        for h in heads:
            h_page = min(max(leaf.start_page, h["page"]), leaf.end_page)
            # Skip headings on the leaf's own start_page - the leaf itself.
            if h_page <= leaf.start_page:
                continue
            title = h["title"].strip()
            if not title:
                continue
            norm = _normalize_title(title)
            if not norm or norm == parent_norm:
                continue
            key = (norm, h_page)
            if key in seen:
                continue
            seen.add(key)
            collected.append({"title": title, "level": leaf.level + 1, "page": h_page})

        if w_end >= leaf.end_page:
            break
        page = max(page + 1, w_end + 1 - config.window_overlap)

    collected.sort(key=lambda h: (h["page"], h["level"]))
    return collected


def _round_half_up(value: float) -> int:
    """Round-half-up (1.4 -> 1, 2.5 -> 3). Python's built-in round() uses
    banker's rounding (2.5 -> 2), which is not what the chunking spec calls for."""
    return math.floor(value + 0.5)


def _fixed_chunk_leaf(leaf: Node, config: Config, verbose: bool) -> bool:
    """Replace `leaf` with fixed-size page-chunk children. Returns True if
    chunking was applied (>= 2 chunks), False if it was a no-op.

    Spec:
      1. N = pages / window_size                      (raw)
      2. N_sub = round_half_up(N)                     (>= 1)
      3. split_size = pages / N_sub                   (raw)
      4. split_size = round_half_up(split_size)       (>= 1)
      5. The LAST chunk takes all remaining pages even if larger than split_size.

    New chunks carry source="window" so summarize_tree treats them like other
    window-derived nodes (asks the LLM for a real title from the page text
    rather than keeping the synthetic 'Pages X-Y' placeholder) and
    `accurate=None` since they bypass verification.
    """
    pages = leaf.end_page - leaf.start_page + 1
    n_sub = max(1, _round_half_up(pages / max(1, config.window_size)))
    if n_sub < 2:
        return False
    split_size = max(1, _round_half_up(pages / n_sub))

    children: List[Node] = []
    cur = leaf.start_page
    for i in range(n_sub):
        is_last = (i == n_sub - 1)
        end = leaf.end_page if is_last else min(leaf.end_page, cur + split_size - 1)
        children.append(Node(
            node_id=f"{leaf.node_id}c{i + 1}",
            title=f"Pages {cur}-{end}",
            level=leaf.level + 1,
            start_page=cur,
            end_page=end,
            source="window",
            accurate=None,
        ))
        cur = end + 1
        if cur > leaf.end_page:
            break

    leaf.children = children
    if verbose:
        print(f"[split] fixed-chunk '{leaf.title[:60]}' ({pages}p) -> "
              f"{len(children)} chunk(s) of ~{split_size}p")
    return True


def split_large_leaves(root: Node, pdf: PDFDocument, config: Config,
                       llm: LLMClient, verbose: bool = True) -> int:
    """Refine oversized leaves into parent + discovered children.

    A leaf qualifies for refinement when its page span exceeds
    `config.effective_large_leaf_pages` and it is not a front-matter node.

    Two passes:
      1. Window heading detector runs over each oversized leaf. If it yields
         >= 2 distinct sub-headings, the leaf becomes their parent.
      2. Any leaf still over the threshold afterwards - either the window
         scan produced < 2 headings, or a discovered child is itself still
         oversized - is replaced by fixed-size page-chunk children per the
         _fixed_chunk_leaf spec.

    New children have `accurate=None` (unverified) since the design skips a
    verify pass on them. Returns the total number of leaves that were split
    (window-derived plus fixed-chunked).
    """
    threshold = config.effective_large_leaf_pages

    # Pass 1: window-scan oversized leaves into semantic sub-headings.
    leaves: List[Node] = [
        n for n in iter_post_order(root)
        if n.is_leaf()
        and not _is_front_matter_node(n)
        and (n.end_page - n.start_page + 1) > threshold
    ]
    if verbose and leaves:
        print(f"[split] window-scanning {len(leaves)} oversized leaf(s) "
              f"(threshold={threshold}p)")

    n_window = 0
    for leaf in leaves:
        span = leaf.end_page - leaf.start_page + 1
        heads = _scan_subheadings(pdf, config, llm, leaf, verbose)
        if len(heads) < 2:
            if verbose:
                print(f"[split] '{leaf.title[:60]}' ({span}p): "
                      f"{len(heads)} sub-heading(s) found, "
                      f"will fall back to fixed chunking")
            continue
        children = build_hierarchy(heads, source="window",
                                   id_prefix=f"{leaf.node_id}s")
        leaf.children = children
        for c in children:
            for d in iter_post_order(c):
                d.accurate = None
        assign_end_pages(leaf.children, leaf.end_page)
        n_window += 1
        if verbose:
            print(f"[split] '{leaf.title[:60]}' ({span}p) -> "
                  f"{len(heads)} sub-heading(s)")

    # Pass 2: any leaf still over the threshold gets fixed-size page chunks.
    # We capture the current leaves into a list to avoid mutating the
    # iteration source; new children added in this pass are smaller than
    # window_size by construction and never need further chunking.
    n_fixed = 0
    for n in list(iter_post_order(root)):
        if not n.is_leaf():
            continue
        if _is_front_matter_node(n):
            continue
        if (n.end_page - n.start_page + 1) <= threshold:
            continue
        if _fixed_chunk_leaf(n, config, verbose):
            n_fixed += 1

    if verbose and (n_window or n_fixed):
        print(f"[split] done: {n_window} window-split, {n_fixed} fixed-chunked")
    return n_window + n_fixed


def _leaf_summary_max_tokens(node: Node, config: Config) -> int:
    """Scale max_tokens for a leaf's summary based on its page span.

    A leaf within the normal range gets the default 4096 budget. Oversized
    leaves (the ones the splitter couldn't subdivide) get a larger budget
    proportional to span / window_size, so the summary has room to enumerate
    the topics inside instead of being squeezed into 3-6 sentences.
    """
    span = node.end_page - node.start_page + 1
    threshold = config.effective_large_leaf_pages
    if span <= threshold:
        return 4096
    units = max(1, math.ceil(span / max(1, config.window_size)))
    budget = config.oversized_leaf_summary_unit * units
    return min(config.oversized_leaf_summary_max, max(4096, budget))


def _uncovered_text(node: Node, pdf: PDFDocument) -> str:
    """Return PDF text from `node`'s page range NOT covered by any child.

    With `assign_end_pages`, children cover [first_child.start_page, node.end_page]
    seamlessly, so the only typical uncovered region is the parent's preamble at
    [node.start_page, first_child.start_page - 1]. Inter-child gaps and trailing
    pages are also handled for robustness.
    """
    if not node.children:
        return ""
    children = sorted(node.children, key=lambda c: c.start_page)
    ranges: List[tuple] = []
    cursor = node.start_page
    for c in children:
        if cursor < c.start_page:
            ranges.append((cursor, c.start_page - 1))
        cursor = max(cursor, c.end_page + 1)
    if cursor <= node.end_page:
        ranges.append((cursor, node.end_page))
    parts = [pdf.text_range(s, e) for s, e in ranges]
    return "\n\n".join(p for p in parts if p)


def summarize_tree(root: Node, pdf: PDFDocument, config: Config, llm: LLMClient,
                   verbose: bool = True) -> None:
    """Fill in `summary`, `text` (and refined titles) for every node, children first.

    Leaf `text` is the full page-range text. Parent `text` is only the *extra*
    text not already held by any descendant (typically the preamble paragraphs
    before the first subsection), so each character of the PDF is stored once.
    Parent summaries are written from children summaries plus that extra text.
    """
    for node in iter_post_order(root):
        needs_title = node.source == "window"
        if node.is_leaf():
            text = pdf.text_range(node.start_page, node.end_page)
            node.text = text
            result = llm.complete_json(
                SUMMARY_SYS,
                leaf_summary_user(node, text, needs_title),
                max_tokens=_leaf_summary_max_tokens(node, config),
            )
        else:
            extra_text = _uncovered_text(node, pdf)
            node.text = extra_text
            children_block = "\n".join(
                f"- {c.title}: {c.summary}" for c in node.children
            )
            result = llm.complete_json(
                SUMMARY_SYS,
                parent_summary_user(node, children_block, needs_title, extra_text),
            )
        if isinstance(result, dict):
            node.summary = str(result.get("summary", "")).strip()
            new_title = str(result.get("title", "")).strip()
            if needs_title and new_title:
                node.title = new_title
        if verbose:
            print(f"[summary] {node.title}")
