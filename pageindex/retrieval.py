"""Retrieval: walk down the most relevant branch, then answer with an LLM."""
from typing import Dict, List, Tuple

from .config import Config
from .llm import LLMClient
from .pdf_utils import PDFDocument
from .prompts import ANSWER_SYS, ROUTING_SYS, answer_user, routing_user
from .tree import Node


def _choose_child(llm: LLMClient, query: str, node: Node) -> Dict:
    """Ask the LLM which child branch to follow from `node` (or to stop)."""
    children_block = "\n".join(
        f"[{i}] {c.title} (pages {c.start_page}-{c.end_page})\n    {c.summary}"
        for i, c in enumerate(node.children)
    )
    result = llm.complete_json(ROUTING_SYS, routing_user(query, node, children_block))
    result = result if isinstance(result, dict) else {}

    action = str(result.get("action", "descend")).strip().lower()
    if action not in ("descend", "stop"):
        action = "descend"
    try:
        idx = int(result.get("child_index"))
    except (TypeError, ValueError):
        idx = None
    return {"action": action, "child_index": idx,
            "reasoning": str(result.get("reasoning", ""))}


def retrieve(root: Node, query: str, config: Config, llm: LLMClient,
             verbose: bool = True) -> Tuple[List[Node], List[Dict]]:
    """Descend the single most relevant branch until a leaf or an explicit stop.

    Returns the path of nodes from root to the selected node, plus the routing
    decision made at each step.
    """
    node = root
    path: List[Node] = [root]
    steps: List[Dict] = []
    depth = 0

    while node.children and depth < config.max_depth:
        decision = _choose_child(llm, query, node)
        steps.append(decision)
        if verbose:
            print(f"[retrieve] at '{node.title}': {decision['action']} "
                  f"- {decision['reasoning'][:120]}")
        if decision["action"] != "descend":
            break
        idx = decision["child_index"]
        if idx is None or not (0 <= idx < len(node.children)):
            break
        node = node.children[idx]
        path.append(node)
        depth += 1

    return path, steps


def answer_query(root: Node, pdf: PDFDocument, query: str, config: Config,
                 llm: LLMClient, verbose: bool = True) -> Dict:
    """Run retrieval, then hand the selected section to the LLM to answer."""
    path, steps = retrieve(root, query, config, llm, verbose=verbose)
    target = path[-1]
    breadcrumb = " > ".join(n.title for n in path)
    context = pdf.text_range(target.start_page, target.end_page)
    if verbose:
        print(f"[answer] reading section '{target.title}' "
              f"(pages {target.start_page}-{target.end_page})")
    answer = llm.complete(
        ANSWER_SYS,
        answer_user(query, target, breadcrumb, context),
        max_tokens=config.answer_max_tokens,
    )
    return {
        "query": query,
        "answer": answer.strip(),
        "breadcrumb": breadcrumb,
        "target": target,
        "path": path,
        "steps": steps,
    }
