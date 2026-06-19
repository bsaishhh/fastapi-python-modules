from __future__ import annotations

from collections import Counter

from modules.resume_ats.extraction.models.text_block import Line, TextBlock

# Bullet-point characters (same list as OpenResume)
BULLET_POINTS = ["⋅", "∙", "🞄", "•", "⦁", "⚫", "●", "⬤", "⚬", "○", "-", "–", "—"]

Y_TOLERANCE = 3.0


class LineGrouper:
    """Stage 2: Group text blocks into lines with adjacent-item merging.

    Port of OpenResume's groupTextItemsIntoLines:
    1. Group blocks by Y proximity into lines.
    2. Sort blocks within each line by X.
    3. Merge adjacent blocks whose gap <= typical char width.
    """

    def group_lines(self, blocks: list[TextBlock]) -> list[Line]:
        if not blocks:
            return []

        # --- step A: sort by reading order (top → left) ---
        sorted_blocks = sorted(blocks, key=lambda b: (round(b.y, 1), b.x))

        # --- step B: group into raw lines by Y proximity ---
        raw_lines: list[list[TextBlock]] = []
        current: list[TextBlock] = [sorted_blocks[0]]

        for block in sorted_blocks[1:]:
            if abs(block.y - current[0].y) <= Y_TOLERANCE:
                current.append(block)
            else:
                raw_lines.append(current)
                current = [block]
        if current:
            raw_lines.append(current)

        # --- step C: compute typical char width for merging ---
        typical_cw = self._typical_char_width(blocks)

        # --- step D: build Lines with adjacent merging ---
        result: list[Line] = []

        for group in raw_lines:
            group.sort(key=lambda b: b.x)
            merged = self._merge_adjacent(group, typical_cw)
            if not merged:
                continue

            line_text = " ".join(b.text for b in merged)
            is_bold = all(b.bold for b in merged)
            alpha_text = "".join(c for c in line_text if c.isalpha())
            is_upper = len(alpha_text) > 2 and alpha_text.isupper()

            result.append(
                Line(
                    text=line_text,
                    blocks=merged,
                    y=merged[0].y,
                    is_bold=is_bold,
                    is_uppercase=is_upper,
                    has_eol=True,
                    fontname=merged[0].fontname,
                    height=merged[0].height,
                )
            )

        return result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _typical_char_width(blocks: list[TextBlock]) -> float:
        """Compute average char width from the most common font+height combo."""
        filtered = [b for b in blocks if b.text.strip()]
        if not filtered:
            return 6.0

        combo_counter: Counter[tuple[str, float]] = Counter()
        for b in filtered:
            combo_counter[(b.fontname, round(b.height, 1))] += len(b.text)

        if not combo_counter:
            return 6.0

        (common_fn, common_h), _ = combo_counter.most_common(1)[0]
        matching = [
            b for b in filtered
            if b.fontname == common_fn and abs(b.height - common_h) < 1.0
        ]
        if not matching:
            return 6.0

        total_w = sum(b.width for b in matching)
        total_c = max(sum(len(b.text) for b in matching), 1)
        return total_w / total_c

    @staticmethod
    def _merge_adjacent(
        blocks: list[TextBlock], typical_cw: float
    ) -> list[TextBlock]:
        """Merge blocks whose gap is ≤ typical char width (OpenResume step 2)."""
        if len(blocks) <= 1:
            return list(blocks)

        merged: list[TextBlock] = [blocks[0]]

        for block in blocks[1:]:
            prev = merged[-1]
            gap = block.x - (prev.x + prev.width)

            if gap <= typical_cw:
                sep = _space_between(prev.text, block.text)
                new_text = prev.text + sep + block.text
                merged[-1] = TextBlock(
                    text=new_text,
                    x=prev.x,
                    y=prev.y,
                    width=(block.x + block.width) - prev.x,
                    height=prev.height,
                    fontname=prev.fontname,
                    bold=prev.bold and block.bold,
                )
            else:
                merged.append(block)

        return merged


def _space_between(left: str, right: str) -> str:
    """Decide whether to insert a space when merging two adjacent fragments."""
    if not left or not right:
        return ""
    if left[-1] in " ,;:|." or left[-1] in BULLET_POINTS:
        return "" if right[0] == " " else " "
    if right[0] in "|," or right[0] in BULLET_POINTS:
        return " " if left[-1] != " " else ""
    return ""
