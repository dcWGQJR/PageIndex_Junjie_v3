"""Answer FinanceBench questions against pre-built PageIndex trees and grade them.

For each row of `financebench_subset_QA.xlsx`:
- Look up the pre-built tree at `trees/<doc_name>.tree.json`.
- Retrieve down the tree and answer the question via the existing pipeline.
- Use an LLM judge to compare the predicted answer to the benchmark answer.
- Append a row to an output xlsx with: <all original columns> + predicted_answer
  + retrieval_path + verdict (T/F/?/SKIP/ERROR) + judge_reason.

Usage
-----
    python online.py
    python online.py --limit 5                   # smoke-test the first 5 rows
    python online.py --out my_results.xlsx
"""
import argparse
import sys
import time
from pathlib import Path

from openpyxl import Workbook, load_workbook

from pageindex import Config, LLMClient
from pageindex.pdf_utils import PDFDocument
from pageindex.retrieval import answer_query
from pageindex.tree import load_tree


JUDGE_SYS = (
    "You compare two answers to a financial question and decide whether they convey "
    "the same factual content. Numeric values with reasonable rounding or unit "
    "differences (e.g. $1,577M vs $1.58B) are equivalent. Yes/No answers must match "
    "direction. If the predicted answer says the document does not contain the "
    "information but the expected answer is a concrete fact, the verdict is F."
)


def judge_prompt(question: str, expected: str, predicted: str) -> str:
    return f"""QUESTION:
{question}

EXPECTED ANSWER (ground truth):
{expected}

PREDICTED ANSWER:
{predicted}

Are these two answers substantively equivalent?
Respond with JSON: {{"verdict": "T" or "F", "reason": "<one short sentence>"}}
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Answer FinanceBench questions against pre-built PageIndex trees."
    )
    parser.add_argument("--qa-file", default="FinanceBench/financebench_subset_QA.xlsx",
                        help="Path to the FinanceBench QA xlsx.")
    parser.add_argument("--trees-dir", default="trees",
                        help="Directory containing <doc_name>.tree.json files.")
    parser.add_argument("--pdfs-dir", default="FinanceBench",
                        help="Directory containing the source PDFs.")
    parser.add_argument("--out", default="financebench_results.xlsx",
                        help="Output xlsx path.")
    parser.add_argument("--limit", type=int,
                        help="Process at most N rows (for smoke testing).")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save the output xlsx every N rows.")
    args = parser.parse_args()

    qa_path = Path(args.qa_file)
    out_path = Path(args.out)
    trees_dir = Path(args.trees_dir)
    pdfs_dir = Path(args.pdfs_dir)

    if not qa_path.is_file():
        print(f"Error: QA file not found: {qa_path}", file=sys.stderr)
        return 1

    wb = load_workbook(qa_path, data_only=True)
    ws = wb.active
    header = [c.value for c in ws[1]]
    required = ["doc_name", "question", "answer"]
    missing = [c for c in required if c not in header]
    if missing:
        print(f"Error: required columns missing from QA file: {missing}", file=sys.stderr)
        return 1
    col = {c: header.index(c) for c in required}

    out_wb = Workbook()
    out_ws = out_wb.active
    out_ws.title = "results"
    out_ws.append(header + ["predicted_answer", "retrieval_path", "verdict", "judge_reason"])

    config = Config()
    llm = LLMClient(config)  # shared client; surfaces a clean error early if key is missing

    n_total = ws.max_row - 1
    n_t = n_f = n_unknown = n_skipped = n_errored = 0

    # Cache loaded trees so subsequent rows for the same doc don't reload.
    tree_cache = {}

    for r, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if args.limit and (r - 1) > args.limit:
            break

        row_data = list(row)
        doc_name = row[col["doc_name"]]
        question = row[col["question"]]
        expected = row[col["answer"]]
        expected_str = "" if expected is None else str(expected).strip()

        if not doc_name or not question or not expected_str:
            n_skipped += 1
            out_ws.append(row_data + ["", "", "SKIP", "missing doc_name/question/answer"])
            continue

        tree_path = trees_dir / f"{doc_name}.tree.json"
        pdf_path = pdfs_dir / f"{doc_name}.pdf"

        if not tree_path.exists():
            n_skipped += 1
            reason = f"no tree at {tree_path}"
            print(f"[{r-1}/{n_total}] SKIP  {doc_name}  ({reason})")
            out_ws.append(row_data + ["", "", "SKIP", reason])
            continue
        if not pdf_path.exists():
            n_skipped += 1
            reason = f"no pdf at {pdf_path}"
            print(f"[{r-1}/{n_total}] SKIP  {doc_name}  ({reason})")
            out_ws.append(row_data + ["", "", "SKIP", reason])
            continue

        if doc_name not in tree_cache:
            tree_cache[doc_name] = load_tree(str(tree_path))
        root = tree_cache[doc_name]

        t0 = time.time()
        pdf = PDFDocument(str(pdf_path))
        try:
            try:
                result = answer_query(root, pdf, str(question), config, llm, verbose=False)
                predicted = result["answer"]
                breadcrumb = result["breadcrumb"]
            except Exception as err:  # noqa: BLE001 - log retrieval failures, keep going
                n_errored += 1
                msg = str(err)[:300]
                print(f"[{r-1}/{n_total}] ERR   {doc_name}: {msg}")
                out_ws.append(row_data + ["", "", "ERROR", msg])
                if (r - 1) % args.save_every == 0:
                    out_wb.save(out_path)
                continue
        finally:
            pdf.close()

        try:
            judged = llm.complete_json(
                JUDGE_SYS, judge_prompt(str(question), expected_str, predicted)
            )
            verdict = str(judged.get("verdict", "")).strip().upper()
            if verdict not in ("T", "F"):
                verdict = "?"
            reason = str(judged.get("reason", "")).strip()
        except Exception as err:  # noqa: BLE001 - record judge failures explicitly
            verdict = "?"
            reason = f"judge failed: {err}"

        if verdict == "T":
            n_t += 1
        elif verdict == "F":
            n_f += 1
        else:
            n_unknown += 1

        dt = time.time() - t0
        q_preview = (str(question)[:60] + "...") if len(str(question)) > 60 else str(question)
        judged_so_far = n_t + n_f
        running = f"running {n_t}/{judged_so_far}" if judged_so_far else "running -"
        print(f"[{r-1}/{n_total}] {verdict}  {doc_name}  ({dt:.1f}s)  [{running}]  q={q_preview!r}")

        out_ws.append(row_data + [predicted, breadcrumb, verdict, reason])

        if (r - 1) % args.save_every == 0:
            out_wb.save(out_path)

    out_wb.save(out_path)

    total_judged = n_t + n_f
    print(f"\n{'=' * 60}")
    print(f"Done.  T={n_t}  F={n_f}  ?={n_unknown}  skipped={n_skipped}  errored={n_errored}")
    if total_judged:
        print(f"Accuracy (T / (T+F)) = {n_t}/{total_judged} = {n_t / total_judged:.1%}")
    print(f"Results saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
