"""Tree construction: from the PDF's embedded TOC, or via sliding-window analysis."""
import os
from typing import Dict, List, Tuple

from .config import Config
from .llm import LLMClient
from .pdf_utils import PDFDocument
from .prompts import HEADING_SYS, SUMMARY_SYS, heading_user, leaf_summary_user, parent_summary_user
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


def build_front_matter(start: int, end: int, config: Config) -> List[Node]:
    """Window the pages before the first content heading into their own nodes.

    These cover title pages, prefaces and the table-of-contents pages so that
    no part of the document is dropped from the tree.
    """
    nodes: List[Node] = []
    page = start
    i = 0
    while page <= end:
        w_end = min(page + config.window_size - 1, end)
        i += 1
        nodes.append(Node(
            node_id=f"f{i}",
            title=f"Front matter (pages {page}-{w_end})",
            level=1,
            start_page=page,
            end_page=w_end,
            source="front_matter",
        ))
        page = w_end + 1
    return nodes


# --------------------------------------------------------------------------
# Path 1: build from the embedded table of contents
# --------------------------------------------------------------------------
def build_from_toc(pdf: PDFDocument, toc: List[Dict], config: Config) -> List[Node]:
    headings = [
        {
            "level": max(1, e["level"]),
            "title": e["title"],
            "page": min(max(1, e["page"]), pdf.page_count),
        }
        for e in toc
    ]
    front = build_front_matter(1, headings[0]["page"] - 1, config)
    content = build_hierarchy(headings, source="toc", id_prefix="t")
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
    front = build_front_matter(1, headings[0]["page"] - 1, config)
    content = build_hierarchy(headings, source="window", id_prefix="w")
    return front + content


# --------------------------------------------------------------------------
# Orchestration: choose a path, then summarize bottom-up
# --------------------------------------------------------------------------
def build_tree(pdf: PDFDocument, config: Config, llm: LLMClient,
               mode: str = "auto", verbose: bool = True) -> Tuple[Node, str]:
    """Build the structural tree (no summaries yet). Returns (root, mode_used)."""
    root = Node(
        node_id="root",
        title=os.path.basename(pdf.path),
        level=0,
        start_page=1,
        end_page=pdf.page_count,
        source="root",
    )

    children = None
    used = None
    if mode in ("auto", "toc"):
        toc = pdf.embedded_toc()
        if len(toc) >= config.toc_min_entries:
            if verbose:
                print(f"[build] using embedded table of contents ({len(toc)} entries)")
            children = build_from_toc(pdf, toc, config)
            used = "toc"
        elif mode == "toc":
            raise ValueError("PDF has no usable embedded table of contents.")

    if children is None:
        if verbose:
            print("[build] analyzing document with a sliding window")
        children = build_from_windows(pdf, config, llm, verbose=verbose)
        used = "window"

    root.children = children
    assign_end_pages(root.children, pdf.page_count)
    return root, used


def summarize_tree(root: Node, pdf: PDFDocument, config: Config, llm: LLMClient,
                   verbose: bool = True) -> None:
    """Fill in `summary` (and refined titles) for every node, children first."""
    for node in iter_post_order(root):
        needs_title = node.source in ("window", "front_matter")
        if node.is_leaf():
            text = pdf.text_range(node.start_page, node.end_page, config.max_chars_per_block)
            result = llm.complete_json(SUMMARY_SYS, leaf_summary_user(node, text, needs_title))
        else:
            children_block = "\n".join(
                f"- {c.title}: {c.summary}" for c in node.children
            )
            result = llm.complete_json(
                SUMMARY_SYS, parent_summary_user(node, children_block, needs_title)
            )
        if isinstance(result, dict):
            node.summary = str(result.get("summary", "")).strip()
            new_title = str(result.get("title", "")).strip()
            if needs_title and new_title:
                node.title = new_title
        if verbose:
            print(f"[summary] {node.title}")
