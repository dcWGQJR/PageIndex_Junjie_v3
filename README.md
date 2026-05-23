# PageIndex

A reasoning-based **tree retrieval** system for PDF documents.

Instead of chunking a PDF and embedding it into a vector store, PageIndex turns
the document into a **hierarchical tree** that mirrors its real structure
(part → chapter → section → subsection). Each node carries a **title** and an
LLM-written **summary**. To answer a question, the system *reasons its way down
the single most relevant branch* of the tree, reads the section it lands on, and
hands that text to an LLM to produce the answer.

## How it works

### 1. Building the tree

```
                ┌─────────────┐
   PDF  ───────▶│  build_tree │
                └──────┬──────┘
            embedded TOC?  ──yes──▶  build_from_toc
                   │
                   no
                   ▼
            build_from_windows  (slide a window over the pages,
                                 ask the LLM to detect headings)
```

- **TOC path** — if the PDF ships with a usable embedded table of contents
  (`>= 3` entries), its outline is used directly. Levels become tree depth.
- **Sliding-window path** — otherwise a window of pages slides across the
  document and the LLM detects section headings and their hierarchy level in
  each window. Detected headings are merged into one structure. If *no*
  headings can be found, the document falls back to flat fixed-size page chunks.
- **Front matter is kept** — every page before the first content heading
  (title page, preface, the TOC pages themselves) becomes its own
  `front_matter` node, so nothing is dropped.
- **Summaries** are generated bottom-up: leaf nodes are summarized from their
  page text; parent nodes are summarized from their children's summaries.

Each node holds: `title`, `summary`, `level`, `start_page`, `end_page`,
`children`.

### 2. Answering a query

Retrieval starts at the root. At every node the LLM is shown the child branches
(title + summary + page range) and picks the **one** most likely to lead to the
answer — or stops if the current node is already the best match. It descends
branch by branch until it reaches a leaf. The selected section's text is then
passed to the LLM, which answers the question and cites page numbers.

## Setup

The virtualenv in this folder is the directory named `.env`.

```powershell
# install dependencies into the existing virtualenv
.env\Scripts\python.exe -m pip install -r requirements.txt

# configure your API key
copy pageindex.env.example pageindex.env
# then edit pageindex.env and paste in your key
```

PageIndex works with **OpenAI** (default — `gpt-4o-mini-2024-07-18`) or
**Anthropic** (`claude-sonnet-4-6`). Switch with `LLM_PROVIDER` in
`pageindex.env`. Override the model with `PAGEINDEX_MODEL`.

## Usage

### Command line

```powershell
# Build a tree and save it
.env\Scripts\python.exe main.py build paper.pdf --out paper.tree.json

# Inspect the tree (add --summaries to see node summaries)
.env\Scripts\python.exe main.py show paper.tree.json --summaries

# Ask a question against a saved tree
.env\Scripts\python.exe main.py ask paper.pdf paper.tree.json "What dataset was used?"

# Build + ask in a single command
.env\Scripts\python.exe main.py query paper.pdf "What dataset was used?"
```

Force a build mode with `--mode toc` or `--mode window` (default `auto`).

### Python API

```python
from pageindex import PageIndex

index = PageIndex()
index.build("paper.pdf")            # mode="auto" | "toc" | "window"
index.save("paper.tree.json")

result = index.answer("What dataset was used?")
print(result["breadcrumb"])         # path taken down the tree
print(result["answer"])

# later, reload without rebuilding
index = PageIndex.load("paper.tree.json", "paper.pdf")
print(index.answer("...")["answer"])
```

## Project layout

| File | Responsibility |
|------|----------------|
| `pageindex/config.py`    | Settings and environment loading |
| `pageindex/pdf_utils.py` | PDF text extraction + embedded-TOC reading |
| `pageindex/tree.py`      | The `Node` type, serialization, rendering |
| `pageindex/builder.py`   | TOC / sliding-window tree construction + summarization |
| `pageindex/retrieval.py` | Branch-walking retrieval + answer generation |
| `pageindex/llm.py`       | Anthropic / OpenAI client wrapper |
| `pageindex/prompts.py`   | All LLM prompts |
| `pageindex/pipeline.py`  | `PageIndex` orchestration class |
| `main.py`                | CLI |

## Notes & limitations

- **Scanned PDFs** (image-only, no text layer) are out of scope — OCR is not
  performed. Pages with no extractable text yield empty summaries.
- Building summarizes every node, so a long book costs one LLM call per
  section. Progress is printed as it runs.
- In sliding-window mode, heading levels are inferred per window; very
  inconsistent document styling can produce uneven nesting.
