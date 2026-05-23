"""Command-line interface for PageIndex.

Examples
--------
    python main.py build  paper.pdf --out paper.tree.json
    python main.py show   paper.tree.json --summaries
    python main.py ask    paper.pdf paper.tree.json "What dataset was used?"
    python main.py query  paper.pdf "What dataset was used?"
"""
import argparse
import sys

from pageindex import Config, PageIndex, load_tree, render_tree


def _print_result(result: dict) -> None:
    print("\n" + "=" * 70)
    print("RETRIEVAL PATH:", result["breadcrumb"])
    print("=" * 70)
    print(result["answer"])
    print("=" * 70)


def cmd_build(args: argparse.Namespace) -> int:
    config = Config()
    if args.window_size:
        config.window_size = args.window_size
    index = PageIndex(config=config)
    index.build(args.pdf, mode=args.mode)
    out = args.out or (args.pdf.rsplit(".", 1)[0] + ".tree.json")
    index.save(out)
    print(f"\nTree built via '{index.mode}' mode and saved to {out}\n")
    print(render_tree(index.root))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    root = load_tree(args.tree)
    print(render_tree(root, show_summary=args.summaries))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    index = PageIndex.load(args.tree, args.pdf)
    _print_result(index.answer(args.query))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    config = Config()
    if args.window_size:
        config.window_size = args.window_size
    index = PageIndex(config=config)
    index.build(args.pdf, mode=args.mode)
    _print_result(index.answer(args.query))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pageindex", description="Reasoning-based tree retrieval over PDFs."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Build a tree index from a PDF.")
    b.add_argument("pdf")
    b.add_argument("--mode", choices=["auto", "toc", "window"], default="auto")
    b.add_argument("--out", help="Path for the saved tree JSON.")
    b.add_argument("--window-size", type=int, help="Pages per sliding window.")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("show", help="Print a saved tree.")
    s.add_argument("tree")
    s.add_argument("--summaries", action="store_true", help="Include node summaries.")
    s.set_defaults(func=cmd_show)

    a = sub.add_parser("ask", help="Answer a query against a saved tree.")
    a.add_argument("pdf")
    a.add_argument("tree")
    a.add_argument("query")
    a.set_defaults(func=cmd_ask)

    q = sub.add_parser("query", help="Build a tree and answer a query in one go.")
    q.add_argument("pdf")
    q.add_argument("query")
    q.add_argument("--mode", choices=["auto", "toc", "window"], default="auto")
    q.add_argument("--window-size", type=int, help="Pages per sliding window.")
    q.set_defaults(func=cmd_query)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (RuntimeError, ValueError, FileNotFoundError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
