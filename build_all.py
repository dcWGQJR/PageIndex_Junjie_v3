"""Batch-build PageIndex trees for every PDF in a directory.

Usage
-----
    # build every PDF under FinanceBench/, write trees/<name>.tree.json
    python build_all.py FinanceBench --out-dir trees --skip-existing

    # only build PDFs that have an embedded TOC (no LLM fallback)
    python build_all.py FinanceBench --skip-on-no-toc

    # smoke-test on the first PDF before committing to the full run
    python build_all.py FinanceBench --limit 1
"""
import argparse
import sys
import time
from pathlib import Path

from pageindex import Config, PageIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-build PageIndex trees.")
    parser.add_argument("input_dir", help="Directory containing PDFs (searched recursively).")
    parser.add_argument("--out-dir", default="trees",
                        help="Directory to write *.tree.json files into.")
    parser.add_argument("--mode",
                        choices=["auto", "toc", "embedded", "printed", "window"],
                        default="auto", help="How to build each tree (default: auto).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip PDFs whose tree.json already exists in --out-dir.")
    parser.add_argument("--skip-on-no-toc", action="store_true",
                        help="Use only the embedded TOC; skip PDFs that lack one.")
    parser.add_argument("--limit", type=int,
                        help="Process at most N PDFs (useful for a smoke test).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-node progress while building.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        print(f"Error: {in_dir} is not a directory.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(in_dir.rglob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    print(f"Found {len(pdfs)} PDF(s) under {in_dir}")
    if not pdfs:
        return 0

    config = Config()
    effective_mode = "toc" if args.skip_on_no_toc else args.mode

    successes: list[str] = []
    skipped: list[str] = []
    failures: list[tuple[str, str]] = []
    log_path = out_dir / "_build_log.txt"

    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n--- batch run {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        for i, pdf_path in enumerate(pdfs, 1):
            out_path = out_dir / (pdf_path.stem + ".tree.json")
            tag = f"[{i}/{len(pdfs)}] {pdf_path.name}"

            if args.skip_existing and out_path.exists():
                print(f"{tag}: already built, skipping.")
                skipped.append(pdf_path.name)
                continue

            print(f"\n{tag}: building (mode={effective_mode})...")
            t0 = time.time()
            try:
                index = PageIndex(config=config)
                index.build(str(pdf_path), mode=effective_mode, verbose=args.verbose)
                index.save(str(out_path))
            except Exception as err:  # noqa: BLE001 - keep batch alive across failures
                line = f"FAIL {pdf_path.name}  error={err}"
                print(line, file=sys.stderr)
                log.write(line + "\n")
                log.flush()
                failures.append((pdf_path.name, str(err)))
                continue

            dt = time.time() - t0
            line = (f"OK   {pdf_path.name}  mode={index.mode}  "
                    f"pages={index.root.end_page}  time={dt:.1f}s  -> {out_path.name}")
            print(line)
            log.write(line + "\n")
            log.flush()
            successes.append(pdf_path.name)

    print(f"\nDone. {len(successes)} built, {len(skipped)} skipped, {len(failures)} failed.")
    print(f"Log: {log_path}")
    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
