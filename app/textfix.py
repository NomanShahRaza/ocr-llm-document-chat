from __future__ import annotations

import re

_MATH = re.compile(r'(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\))')


def literal_newlines(text: str, replacement: str = '<br>') -> str:
    """Replace the two-character sequence backslash-n that VLMs emit inside table cells.

    Math spans are left untouched because LaTeX commands such as \\nu, \\neq and \\nabla start with it.
    """
    if not text or '\\n' not in text:
        return text
    parts = _MATH.split(text)
    for index in range(0, len(parts), 2):
        parts[index] = parts[index].replace('\\n', replacement)
    return ''.join(parts)
