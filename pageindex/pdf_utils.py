"""PDF access: page text extraction and embedded table-of-contents reading."""
import re
from typing import Dict, List, Optional

import fitz  # PyMuPDF


# A "Item 1A. Risk Factors .......... 12" style entry (single-line dot-leader TOC).
_LEADER_LINE = re.compile(r"\.{3,}\s*\d{1,4}\s*$")
# A line that is just a page-number cell, e.g. "12" on its own line — the
# tell-tale sign of a column-layout TOC where titles and page numbers are in
# separate columns. Body prose effectively never produces these.
_PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$")


class PDFDocument:
    """Wraps a PyMuPDF document with cached, 1-indexed page access."""

    def __init__(self, path: str):
        self.path = path
        self.doc = fitz.open(path)
        self.page_count: int = self.doc.page_count
        self._cache: Dict[int, str] = {}

    def page_text(self, page_number: int) -> str:
        """Return the plain text of a 1-indexed page (cached)."""
        if page_number in self._cache:
            return self._cache[page_number]
        if page_number < 1 or page_number > self.page_count:
            return ""
        text = self.doc[page_number - 1].get_text("text")
        self._cache[page_number] = text
        return text

    def text_range(self, start: int, end: int, max_chars: Optional[int] = None) -> str:
        """Concatenate pages `start..end` (inclusive) with `[page N]` markers."""
        start = max(1, start)
        end = min(self.page_count, end)
        parts = [f"[page {p}]\n{self.page_text(p)}" for p in range(start, end + 1)]
        text = "\n\n".join(parts)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]..."
        return text

    def embedded_toc(self) -> List[Dict]:
        """Return the PDF's built-in outline as `{level, title, page}` dicts.

        Entries without a real page destination are skipped. Empty list means
        the PDF has no usable table of contents.
        """
        toc: List[Dict] = []
        for level, title, page in self.doc.get_toc(simple=True):
            if page < 1:
                continue
            toc.append({
                "level": int(level),
                "title": (title or "").strip() or "Untitled",
                "page": int(page),
            })
        return toc

    def find_printed_toc_pages(self, max_scan: int = 25, max_pages: int = 3,
                               start_page: int = 1) -> List[int]:
        """Locate the page(s) of a printed (in-body) table of contents.

        Scans pages in [start_page, start_page + max_scan - 1] (clamped to the
        document range). A page is classified as TOC-like when it shows one
        of these structural patterns (header phrases are deliberately ignored -
        "Table of Contents" is often a running header on every page in
        financial filings):
          - several dot-leader lines like ".......... 12", or
          - many lines that are just a page-number cell (column-layout TOC)
            *and* a short median line length (rules out body prose).

        We pick the first TOC-like page in the scanned range as the anchor
        and extend forward up to `max_pages` total only while following pages
        are also TOC-like. Returns the page numbers (1-indexed) or [] if no
        TOC page was found in the scanned range. The `start_page` parameter
        lets callers search for additional TOC blocks beyond an initial scan
        (e.g. mid-document TOCs in long filings).
        """
        is_toc: List[bool] = []
        first = max(1, start_page)
        last = min(first + max_scan - 1, self.page_count)
        if first > last:
            return []
        for p in range(first, last + 1):
            text = self.page_text(p)
            if not text.strip():
                is_toc.append(False)
                continue
            lines = [ln for ln in text.splitlines() if ln.strip()]
            leaders = sum(1 for ln in lines if _LEADER_LINE.search(ln))
            pure_nums = sum(1 for ln in lines if _PAGE_NUM_LINE.match(ln))
            sorted_lens = sorted(len(ln) for ln in lines)
            median_len = sorted_lens[len(sorted_lens) // 2] if sorted_lens else 0
            is_toc.append(
                leaders >= 5
                or (pure_nums >= 5 and median_len < 60)
            )

        try:
            idx = is_toc.index(True)
        except ValueError:
            return []

        anchor_page = first + idx
        pages = [anchor_page]
        for step in range(1, max_pages):
            i = idx + step
            if i >= len(is_toc) or not is_toc[i]:
                break
            pages.append(first + i)
        return pages

    def close(self) -> None:
        self.doc.close()

    def __enter__(self) -> "PDFDocument":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
