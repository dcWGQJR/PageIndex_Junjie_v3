"""High-level entry point that ties building, storage and retrieval together."""
import os
from typing import Dict, Optional

from .builder import build_tree, summarize_tree
from .config import Config
from .llm import LLMClient
from .pdf_utils import PDFDocument
from .retrieval import answer_query
from .tree import Node, load_tree, save_tree
from .verify import verify_and_repair


class PageIndex:
    """Build a reasoning-based tree index of a PDF and answer queries over it."""

    def __init__(self, config: Optional[Config] = None,
                 llm: Optional[LLMClient] = None):
        self.config = config or Config()
        self._llm = llm
        self.root: Optional[Node] = None
        self.pdf_path: Optional[str] = None
        self.mode: Optional[str] = None
        self.metrics: Optional[Dict] = None

    @property
    def llm(self) -> LLMClient:
        """Lazily create the LLM client (so loading a saved tree needs no key)."""
        if self._llm is None:
            self._llm = LLMClient(self.config)
        return self._llm

    # -- building -----------------------------------------------------------
    def build(self, pdf_path: str, mode: str = "auto", verbose: bool = True) -> Node:
        """Construct the tree for a PDF. `mode` is "auto", "toc" or "window".

        Runs in order: build_tree -> verify_and_repair -> summarize_tree. Verify
        comes BEFORE summarize so that `node.text` and `node.summary` are
        written against the final post-repair page ranges (otherwise they
        capture the build-time ranges and go stale when verify moves leaves or
        `refine_end_pages` bumps an end_page). Verification metrics are stored
        on `self.metrics`.
        """
        if mode not in ("auto", "toc", "window"):
            raise ValueError("mode must be 'auto', 'toc' or 'window'")
        self.pdf_path = os.path.abspath(pdf_path)
        pdf = PDFDocument(self.pdf_path)
        try:
            root, used = build_tree(pdf, self.config, self.llm, mode=mode, verbose=verbose)
            self.metrics = verify_and_repair(root, pdf, llm=self.llm, verbose=verbose)
            summarize_tree(root, pdf, self.config, self.llm, verbose=verbose)
        finally:
            pdf.close()
        self.root, self.mode = root, used
        return root

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
