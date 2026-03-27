"""Compatibility shim for environments without the ftfy package.

The vendored CLIP tokenizer only needs ``fix_text``. Returning the original
text is sufficient for the prompt strings used in this project.
"""


def fix_text(text: str) -> str:
    return text
