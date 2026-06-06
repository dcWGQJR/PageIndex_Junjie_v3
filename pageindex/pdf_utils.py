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

# PyMuPDF span flag bit 4 (= 16) is set when the run is rendered bold.
_FLAG_BOLD = 1 << 4
# A line whose dominant span exceeds the page's median body font size by this
# many points is treated as visually heading-like. Two points is a typical
# step between body (10-11pt) and a heading (12-14pt) in financial filings.
_HEADING_SIZE_DELTA = 2.0
# Lines longer than this are paragraphs, not headings, even if bold.
_HEADING_MAX_LEN = 200
# Lines shorter than this are likely fragments (page numbers, footers).
_HEADING_MIN_LEN = 3


def _line_text(line: dict) -> str:
    return "".join(s.get("text", "") for s in line.get("spans", []))


def _dominant_span(line: dict) -> Optional[dict]:
    """Span that contributes the most characters to a line - used so a stray
    short bold span in a mostly-body line doesn't mis-classify it."""
    spans = [s for s in line.get("spans", []) if s.get("text")]
    if not spans:
        return None
    return max(spans, key=lambda s: len(s.get("text", "")))


def _page_median_body_size(blocks: List[dict]) -> float:
    """Character-weighted median font size on the page. Used as the body-text
    baseline so heading detection is robust to docs that use 9pt or 12pt body."""
    sizes: List[float] = []
    for b in blocks:
        if b.get("type", 0) != 0:
            continue
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                t = sp.get("text", "")
                if not t.strip():
                    continue
                size = float(sp.get("size", 0))
                sizes.extend([size] * len(t))
    if not sizes:
        return 0.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def _line_is_heading_like(line: dict, body_size: float) -> bool:
    text = _line_text(line).strip()
    if not (_HEADING_MIN_LEN <= len(text) <= _HEADING_MAX_LEN):
        return False
    dom = _dominant_span(line)
    if dom is None:
        return False
    is_bold = bool(int(dom.get("flags", 0)) & _FLAG_BOLD)
    is_large = body_size > 0 and float(dom.get("size", body_size)) >= body_size + _HEADING_SIZE_DELTA
    return is_bold or is_large


class PDFDocument:
    """Wraps a PyMuPDF document with cached, 1-indexed page access."""

    def __init__(self, path: str):
        self.path = path
        self.doc = fitz.open(path)
        self.page_count: int = self.doc.page_count
        self._cache: Dict[int, str] = {}
        self._annotated_cache: Dict[int, str] = {}

    def page_text(self, page_number: int) -> str:
        """Return the plain text of a 1-indexed page (cached)."""
        if page_number in self._cache:
            return self._cache[page_number]
        if page_number < 1 or page_number > self.page_count:
            return ""
        text = self.doc[page_number - 1].get_text("text")
        self._cache[page_number] = text
        return text

    def page_text_annotated(self, page_number: int) -> str:
        """Page text with heading-like lines prefixed by `# `.

        Uses PyMuPDF's `get_text("dict")` rich layout to recover font size
        and bold flag information that plain `get_text("text")` loses. A line
        is annotated when its dominant span is bold OR its font size exceeds
        the page's median body size by at least 2 points. The marker gives
        the sliding-window heading detector a strong visual signal even when
        the prose itself doesn't read as a heading.
        """
        if page_number in self._annotated_cache:
            return self._annotated_cache[page_number]
        if page_number < 1 or page_number > self.page_count:
            return ""
        data = self.doc[page_number - 1].get_text("dict")
        blocks = data.get("blocks", [])
        body_size = _page_median_body_size(blocks)
        out: List[str] = []
        for b in blocks:
            if b.get("type", 0) != 0:
                continue
            for ln in b.get("lines", []):
                text = _line_text(ln).rstrip()
                if not text.strip():
                    continue
                if _line_is_heading_like(ln, body_size):
                    out.append(f"# {text.strip()}")
                else:
                    out.append(text)
        result = "\n".join(out)
        self._annotated_cache[page_number] = result
        return result

    def text_range(self, start: int, end: int, max_chars: Optional[int] = None) -> str:
        """Concatenate pages `start..end` (inclusive) with `[page N]` markers."""
        start = max(1, start)
        end = min(self.page_count, end)
        parts = [f"[page {p}]\n{self.page_text(p)}" for p in range(start, end + 1)]
        text = "\n\n".join(parts)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]..."
        return text

    def text_range_annotated(self, start: int, end: int,
                             max_chars: Optional[int] = None) -> str:
        """Like `text_range`, but each page's text is annotated with `# `
        markers on heading-like lines (see `page_text_annotated`). Used by
        the sliding-window heading detector so the LLM sees font/bold cues
        that plain text extraction discards."""
        start = max(1, start)
        end = min(self.page_count, end)
        parts = [f"[page {p}]\n{self.page_text_annotated(p)}" for p in range(start, end + 1)]
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
