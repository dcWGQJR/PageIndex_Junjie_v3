"""PDF access: page text extraction and embedded table-of-contents reading."""
from typing import Dict, List, Optional

import fitz  # PyMuPDF


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

    def close(self) -> None:
        self.doc.close()

    def __enter__(self) -> "PDFDocument":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
