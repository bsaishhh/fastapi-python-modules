from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextBlock:
    """A single text fragment extracted from a PDF page."""

    text: str
    x: float
    y: float
    width: float
    height: float
    fontname: str
    bold: bool


@dataclass
class Line:
    """A logical line of text composed of one or more TextBlocks."""

    text: str
    blocks: list[TextBlock]
    y: float
    is_bold: bool = False
    is_uppercase: bool = False
    has_eol: bool = True
    fontname: str = ""
    height: float = 0.0
