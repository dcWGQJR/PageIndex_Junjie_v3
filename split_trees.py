"""Post-process saved tree.json files: split oversized leaves via window scan.

For each `*.tree.json` in `--in-dir`:
  1. Load the tree.
  2. Locate the matching PDF in `--pdf-dir` (filename stem must match).
  3. Walk leaves whose page span exceeds the threshold; for each, run the
     sliding-window heading detector to discover sub-headings the source TOC
     didn't list, and graft them as children.
  4. Re-summarize the affected subtrees (the new children, plus the
     now-parent so its summary reflects what's underneath).
  5. Save the updated tree to `--out-dir`.

Trees with no oversized leaves are copied through unchanged.

Usage
-----
    python split_trees.py --in-dir trees --out-dir trees_split --pdf-dir FinanceBench
    python split_trees.py --in-dir trees --out-dir trees_split --pdf-dir FinanceBench \
        --threshold 30 --limit 5
"""
import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from pageindex import Config, LLMClient
from pageindex.builder import (
    _leaf_summary_max_tokens,
    split_large_leaves,
)
from pageindex.pdf_utils import PDFDocument
from pageindex.prompts import SUMMARY_SYS, leaf_summary_user, parent_summary_user
from pageindex.tree import Node, iter_post_order, load_tree, save_tree
from pageindex.builder import _uncovered_text


def _has_oversized_leaf(root: Node, threshold: int) -> bool:
    for n in iter_post_order(root):
        if not n.is_leaf():
            continue
        if n.node_id.startswith("f") or n.source in ("preface", "toc", "front_matter"):
            continue
        if (n.end_page - n.start_page + 1) > threshold:
            return True
    return False


def _resummarize_subtree(node: Node, pdf: PDFDocument, config: Config,
                        llm: LLMClient, verbose: bool) -> None:
    """Bottom-up re-summarize: every leaf gets text+summary; every parent
    gets a parent summary from its children's summaries plus uncovered text.

    Used after a leaf has been promoted to a parent: regenerates summaries
    for the new children and overwrites the original leaf's summary with a
    parent-style summary that reflects the new structure.
    """
    for n in iter_post_order(node):
        if n.is_leaf():
            text = pdf.text_range(n.start_page, n.end_page)
            n.text = text
            try:
                result = llm.complete_json(
                    SUMMARY_SYS,
                    leaf_summary_user(n, text, needs_title=False),
                    max_tokens=_leaf_summary_max_tokens(n, config),
                )
            except Exception as err:  # noqa: BLE001 - degrade to keeping old summary
                if verbose:
                    print(f"[resummarize] leaf '{n.title[:50]}' error: {err}")
                continue
        else:
            extra_text = _uncovered_text(n, pdf)
            n.text = extra_text
            block = "\n".join(f"- {c.title}: {c.summary}" for c in n.children)
            try:
                result = llm.complete_json(
                    SUMMARY_SYS,
                    parent_summary_user(n, block, needs_title=False,
                                        extra_text=extra_text),
                )
            except Exception as err:  # noqa: BLE001
                if verbose:
                    print(f"[resummarize] parent '{n.title[:50]}' error: {err}")
                continue
        if isinstance(result, dict):
            new_summary = str(result.get("summary", "")).strip()
            if new_summary:
                n.summary = new_summary
        if verbose:
            print(f"[resummarize] {n.title[:70]}")


def _process_one(tree_path: Path, out_path: Path, pdf_dir: Path,
                 config: Config, llm: LLMClient, verbose: bool) -> str:
    """Process one tree. Returns one of: "split", "copied", "no-pdf", "error"."""
    root = load_tree(str(tree_path))
    threshold = config.effective_large_leaf_pages

    if not _has_oversized_leaf(root, threshold):
        shutil.copyfile(tree_path, out_path)
        return "copied"

    pdf_name = tree_path.name.replace(".tree.json", ".pdf")
    candidates = list(pdf_dir.rglob(pdf_name))
    if not candidates:
        # Copy through so the output dir stays a complete mirror.
        shutil.copyfile(tree_path, out_path)
        return "no-pdf"

    pdf = PDFDocument(str(candidates[0]))
    try:
        # Snapshot the leaves we expect to operate on so we can re-summarize them after.
        # `split_large_leaves` mutates the tree in place; we capture references
        # before the mutation by node_id.
        targets: List[Node] = [
            n for n in iter_post_order(root)
            if n.is_leaf()
            and not n.node_id.startswith("f")
            and n.source not in ("preface", "toc", "front_matter")
            and (n.end_page - n.start_page + 1) > threshold
        ]
        n_split = split_large_leaves(root, pdf, config, llm, verbose=verbose)
        if n_split == 0:
            # Window scan found no usable headings; the longer-summary
            # fallback still applies to these leaves.
            for leaf in targets:
                _resummarize_subtree(leaf, pdf, config, llm, verbose=verbose)
        else:
            # Re-summarize each split subtree (the new children and the
            # now-parent). Targets that didn't actually split (children
            # empty) just get a fresh leaf summary with the larger budget.
            for leaf in targets:
                _resummarize_subtree(leaf, pdf, config, llm, verbose=verbose)
    finally:
        pdf.close()

    save_tree(root, str(out_path))
    return "split"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split oversized leaves in already-saved tree.json files."
    )
    parser.add_argument("--in-dir", default="trees",
                        help="Directory of input *.tree.json files.")
    parser.add_argument("--out-dir", default="trees_split",
                        help="Directory to write processed trees into.")
    parser.add_argument("--pdf-dir", default="FinanceBench",
                        help="Directory containing the source PDFs "
                             "(searched recursively by filename stem).")
    parser.add_argument("--threshold", type=int,
                        help="Override Config.large_leaf_pages "
                             "(0/unset means 2 * window_size).")
    parser.add_argument("--limit", type=int,
                        help="Process at most N trees (smoke test).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    pdf_dir = Path(args.pdf_dir)

    if not in_dir.is_dir():
        print(f"Error: {in_dir} is not a directory.", file=sys.stderr)
        return 1
    if not pdf_dir.is_dir():
        print(f"Error: {pdf_dir} is not a directory.", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    if args.threshold is not None:
        config.large_leaf_pages = args.threshold
    llm = LLMClient(config)

    trees = sorted(in_dir.glob("*.tree.json"))
    if args.limit:
        trees = trees[: args.limit]
    print(f"Found {len(trees)} tree(s) in {in_dir} "
          f"(threshold={config.effective_large_leaf_pages}p)")
    if not trees:
        return 0

    counts: Dict[str, int] = {"split": 0, "copied": 0, "no-pdf": 0, "error": 0}
    for i, tree_path in enumerate(trees, 1):
        out_path = out_dir / tree_path.name
        tag = f"[{i}/{len(trees)}] {tree_path.name}"
        t0 = time.time()
        try:
            status = _process_one(tree_path, out_path, pdf_dir, config, llm,
                                  args.verbose)
        except Exception as err:  # noqa: BLE001 - keep batch alive
            counts["error"] += 1
            print(f"{tag}: ERROR {err}", file=sys.stderr)
            continue
        counts[status] += 1
        dt = time.time() - t0
        marker = {"split": "SPLIT", "copied": "COPY", "no-pdf": "NO-PDF",
                  "error": "ERR"}[status]
        print(f"{tag}: {marker}  ({dt:.1f}s)")

    print(f"\nDone. split={counts['split']}  copied={counts['copied']}  "
          f"no-pdf={counts['no-pdf']}  errored={counts['error']}")
    print(f"Output: {out_dir}")
    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())