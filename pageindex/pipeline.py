"""High-level entry point that ties building, storage and retrieval together."""
import os
import time
from typing import Dict, Optional

from .builder import build_tree, split_large_leaves, summarize_tree
from .config import Config
from .llm import LLMClient
from .pdf_utils import PDFDocument
from .retrieval import answer_query
from .tree import Node, load_tree, save_tree
from .verify import verify_and_repair


class PageIndex:
    """Build a reasoning-based tree index of a PDF and answer queries over it.

    Building is exposed in phases so callers can decide what work to skip
    when a tree won't be saved (e.g. accuracy below threshold). The
    convenience method `build()` runs all phases in one call.

    Phases (call in order; `summarize` must come last before save):
        index.build_structure(pdf_path, mode)   # build_tree + verify_and_repair
        index.split_oversized()                 # optional: refine large leaves
        index.summarize()                       # write node.text + node.summary
        index.save(path)
    """

    def __init__(self, config: Optional[Config] = None,
                 llm: Optional[LLMClient] = None):
        self.config = config or Config()
        self._llm = llm
        self.root: Optional[Node] = None
        self.pdf_path: Optional[str] = None
        self.mode: Optional[str] = None
        self.metrics: Optional[Dict] = None
        self.timings: Optional[Dict[str, float]] = None
        self._pdf: Optional[PDFDocument] = None

    @property
    def llm(self) -> LLMClient:
        """Lazily create the LLM client (so loading a saved tree needs no key)."""
        if self._llm is None:
            self._llm = LLMClient(self.config)
        return self._llm

    def close(self) -> None:
        """Release the PDF handle if one is open. Safe to call multiple times."""
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    def __enter__(self) -> "PageIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- phased building ----------------------------------------------------
    def build_structure(self, pdf_path: str, mode: str = "auto",
                        verbose: bool = True) -> Node:
        """Phase 1: build the tree and run the verify-and-repair fix loop.

        Stores `self.root`, `self.mode`, `self.metrics`, opens and retains a
        PDFDocument on `self._pdf`. The PDF remains open until `summarize`
        runs or `close()` is called; the cached page text is reused across
        phases.
        """
        if mode not in ("auto", "toc", "window"):
            raise ValueError("mode must be 'auto', 'toc' or 'window'")
        self.pdf_path = os.path.abspath(pdf_path)
        self._pdf = PDFDocument(self.pdf_path)
        self.timings = self.timings or {}

        t0 = time.perf_counter()
        root, used = build_tree(self._pdf, self.config, self.llm, mode=mode,
                                verbose=verbose)
        self.timings["build"] = time.perf_counter() - t0
        if verbose:
            print(f"[time] build_tree: {self.timings['build']:.1f}s")

        t0 = time.perf_counter()
        self.metrics = verify_and_repair(root, self._pdf, self.config,
                                         self.llm, verbose=verbose)
        self.timings["verify"] = time.perf_counter() - t0
        if verbose:
            print(f"[time] verify_and_repair: {self.timings['verify']:.1f}s")

        self.root, self.mode = root, used
        return root

    def split_oversized(self, verbose: bool = True) -> int:
        """Phase 2 (optional): refine oversized leaves with the window scan."""
        if self.root is None or self._pdf is None:
            raise RuntimeError("Call build_structure() before split_oversized().")
        if not self.config.split_large_leaves:
            return 0
        self.timings = self.timings or {}
        t0 = time.perf_counter()
        split = split_large_leaves(self.root, self._pdf, self.config, self.llm,
                                   verbose=verbose)
        self.timings["split"] = time.perf_counter() - t0
        if verbose:
            print(f"[time] split_large_leaves: {self.timings['split']:.1f}s "
                  f"({split} leaf/leaves split)")
        return split

    def summarize(self, verbose: bool = True) -> None:
        """Phase 3: fill in node.text and node.summary bottom-up. Closes the PDF."""
        if self.root is None or self._pdf is None:
            raise RuntimeError("Call build_structure() before summarize().")
        self.timings = self.timings or {}
        t0 = time.perf_counter()
        try:
            summarize_tree(self.root, self._pdf, self.config, self.llm,
                           verbose=verbose)
        finally:
            self.close()
        self.timings["summarize"] = time.perf_counter() - t0
        if verbose:
            print(f"[time] summarize_tree: {self.timings['summarize']:.1f}s")

    # -- convenience wrapper ------------------------------------------------
    def build(self, pdf_path: str, mode: str = "auto",
              split: Optional[bool] = None, verbose: bool = True) -> Node:
        """Construct the tree end-to-end: structure + (optional split) + summarize.

        `split=None` defers to `config.split_large_leaves`. Set explicitly to
        skip the split even when the config enables it (e.g. when the caller
        doesn't intend to save the tree).
        """
        self.build_structure(pdf_path, mode=mode, verbose=verbose)
        if split if split is not None else self.config.split_large_leaves:
            self.split_oversized(verbose=verbose)
        self.summarize(verbose=verbose)
        return self.root

    # -- persistence --------------------------------------------------------
    def save(self, tree_path: str) -> None:
        if self.root is None:
            raise RuntimeError("Nothing to save - build an index first.")
        save_tree(self.root, tree_path)

    @classmethod
    def load(cls, tree_path: str, pdf_path: str,
             config: Optional[Config] = None) -> "PageIndex":
        """Reload a previously saved tree, paired with its source PDF."""
        index = cls(config=config)
        index.root = load_tree(tree_path)
        index.pdf_path = os.path.abspath(pdf_path)
        return index

    # -- querying -----------------------------------------------------------
    def answer(self, query: str, verbose: bool = True) -> Dict:
        """Retrieve the most relevant section and answer the query over it."""
        if self.root is None or self.pdf_path is None:
            raise RuntimeError("Build or load an index before querying.")
        pdf = PDFDocument(self.pdf_path)
        try:
            return answer_query(self.root, pdf, query, self.config, self.llm,
                                verbose=verbose)
        finally:
            pdf.close()