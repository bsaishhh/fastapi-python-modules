from __future__ import annotations

import io
import logging
import re

import pdfplumber

from modules.resume_ats.extraction.models.text_block import TextBlock

logger = logging.getLogger(__name__)

_BOLD_RE = re.compile(r"(bold|black|heavy)", re.IGNORECASE)


def _is_bold_font(fontname: str) -> bool:
    return bool(_BOLD_RE.search(fontname))


class PDFTextExtractor:
    """Stage 1: Extract character-level TextBlocks with positional/font metadata.

    Uses pdfplumber's per-character data so we can detect word boundaries by
    measuring inter-character gaps against the typical character width.
    This avoids the merging problem seen with extract_words().
    """

    def extract(self, file_bytes: bytes) -> list[TextBlock]:
        blocks: list[TextBlock] = []

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                chars = page.chars or []
                if not chars:
                    blocks.extend(self._text_fallback(page))
                    continue

                # Sort chars by reading order: top → x0
                sorted_chars = sorted(
                    chars,
                    key=lambda c: (round(c.get("top", 0), 1), c.get("x0", 0)),
                )

                # Step 1: group into raw lines by Y proximity
                raw_lines = self._group_chars_into_lines(sorted_chars)

                # Step 2: compute typical char width for this page
                typical_cw = self._page_typical_char_width(raw_lines)

                # Step 3: within each line, split into word blocks by gap
                for line_chars in raw_lines:
                    line_chars.sort(key=lambda c: c.get("x0", 0))
                    word_groups = self._split_into_words(line_chars, typical_cw)
                    for wg in word_groups:
                        block = self._chars_to_block(wg)
                        if block.text:
                            blocks.append(block)

        logger.debug("Extracted %d text blocks from PDF", len(blocks))
        return blocks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text_fallback(page) -> list[TextBlock]:
        """Last-resort: use extract_text() when no char data is available."""
        blocks: list[TextBlock] = []
        text = page.extract_text() or ""
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if line:
                blocks.append(
                    TextBlock(
                        text=line,
                        x=0.0,
                        y=float(i) * 14.0,
                        width=float(len(line)) * 6.0,
                        height=12.0,
                        fontname="",
                        bold=False,
                    )
                )
        return blocks

    @staticmethod
    def _group_chars_into_lines(sorted_chars: list[dict]) -> list[list[dict]]:
        """Group sorted characters into lines using Y proximity (tolerance 2pt)."""
        if not sorted_chars:
            return []

        y_tolerance = 2.0
        lines: list[list[dict]] = []
        current: list[dict] = [sorted_chars[0]]

        for ch in sorted_chars[1:]:
            if abs(ch.get("top", 0) - current[0].get("top", 0)) <= y_tolerance:
                current.append(ch)
            else:
                lines.append(current)
                current = [ch]
        if current:
            lines.append(current)

        return lines

    @staticmethod
    def _page_typical_char_width(raw_lines: list[list[dict]]) -> float:
        """Estimate typical character width from the page's characters."""
        widths: list[float] = []
        for line_chars in raw_lines:
            for ch in line_chars:
                w = ch.get("width", 0)
                if w > 0:
                    widths.append(w)
        if not widths:
            return 6.0
        # Use median to be robust against outliers
        widths.sort()
        return widths[len(widths) // 2]

    @staticmethod
    def _split_into_words(
        line_chars: list[dict], typical_cw: float
    ) -> list[list[dict]]:
        """Split a line's characters into word groups based on gap > 0.5 * typical char width."""
        if not line_chars:
            return []

        gap_threshold = typical_cw * 0.5  # half a char width = word boundary
        words: list[list[dict]] = []
        current: list[dict] = [line_chars[0]]

        for ch in line_chars[1:]:
            prev_end = current[-1].get("x1", current[-1].get("x0", 0))
            curr_start = ch.get("x0", 0)
            gap = curr_start - prev_end

            # Skip space characters
            if ch.get("text", "").strip() == "" and gap > gap_threshold:
                if current:
                    words.append(current)
                    current = []
                continue

            if gap > gap_threshold and ch.get("text", "").strip():
                words.append(current)
                current = [ch]
            else:
                current.append(ch)

        if current:
            words.append(current)

        return words

    @staticmethod
    def _chars_to_block(chars: list[dict]) -> TextBlock:
        text = "".join(c.get("text", "") for c in chars).strip()
        if not text:
            return TextBlock(text="", x=0, y=0, width=0, height=0, fontname="", bold=False)
        x0 = min(c.get("x0", 0) for c in chars)
        x1 = max(c.get("x1", 0) for c in chars)
        top = min(c.get("top", 0) for c in chars)
        height = max(c.get("height", 10) for c in chars)
        fontname = chars[0].get("fontname", "")
        bold = any(_is_bold_font(c.get("fontname", "")) for c in chars)
        return TextBlock(
            text=text,
            x=float(x0),
            y=float(top),
            width=float(x1 - x0),
            height=float(height),
            fontname=fontname,
            bold=bold,
        )
