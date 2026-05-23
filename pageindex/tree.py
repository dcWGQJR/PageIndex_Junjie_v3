"""The document tree: the Node type and (de)serialization helpers."""
import json
from dataclasses import dataclass, field
from typing import Iterator, List


@dataclass
class Node:
    """One section of the document.

    `start_page`/`end_page` are 1-indexed and inclusive. A node's range covers
    all of its descendants. `source` records how the node was created:
    "root", "toc", "window" or "front_matter".
    """

    node_id: str
    title: str
    level: int
    start_page: int
    end_page: int
    summary: str = ""
    source: str = ""
    children: List["Node"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return not self.children

    def add_child(self, child: "Node") -> None:
        self.children.append(child)

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "level": self.level,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "summary": self.summary,
            "source": self.source,
            "children": [c.to_dict() for c in self.children],
        }

    @staticmethod
    def from_dict(d: dict) -> "Node":
        node = Node(
            node_id=d["node_id"],
            title=d["title"],
            level=d["level"],
            start_page=d["start_page"],
            end_page=d["end_page"],
            summary=d.get("summary", ""),
            source=d.get("source", ""),
        )
        node.children = [Node.from_dict(c) for c in d.get("children", [])]
        return node


def iter_post_order(node: Node) -> Iterator[Node]:
    """Yield every node with children before their parent (parent last)."""
    for child in node.children:
        yield from iter_post_order(child)
    yield node


def save_tree(root: Node, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(root.to_dict(), fh, indent=2, ensure_ascii=False)


def load_tree(path: str) -> Node:
    with open(path, "r", encoding="utf-8") as fh:
        return Node.from_dict(json.load(fh))


def render_tree(node: Node, show_summary: bool = False, _indent: int = 0) -> str:
    """Produce an indented, human-readable view of the tree."""
    pad = "  " * _indent
    lines = [f"{pad}- {node.title}  (p.{node.start_page}-{node.end_page})"]
    if show_summary and node.summary:
        lines.append(f"{pad}    {node.summary}")
    for child in node.children:
        lines.append(render_tree(child, show_summary, _indent + 1))
    return "\n".join(lines)
