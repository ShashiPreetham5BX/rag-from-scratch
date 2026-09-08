"""
Preset cleaning pipelines.

Three configurations of increasing aggressiveness. Phase 15 compares them
empirically. The hypothesis worth testing: cleaning improves retrieval up
to a point, after which it destroys signal.
"""

from rag.preprocessing.base import CleaningPipeline
from rag.preprocessing.steps import (
    CollapseNewlines,
    CollapseSpaces,
    FixHyphenation,
    NormalizePunctuation,
    NormalizeWhitespace,
    RemoveControlChars,
    UnicodeNormalize,
)


def minimal_pipeline() -> CleaningPipeline:
    """Whitespace only. Near-baseline: what happens with almost no cleaning."""
    return CleaningPipeline([
        CollapseSpaces(),
    ])


def standard_pipeline() -> CleaningPipeline:
    """Default. Fixes representation issues without altering content.

    Order is deliberate: Unicode normalization first, because later
    regex steps assume normalized input.
    """
    return CleaningPipeline([
        UnicodeNormalize("NFKC"),
        RemoveControlChars(),
        NormalizeWhitespace(),
        FixHyphenation(),
        CollapseNewlines(unwrap_single=True),
        CollapseSpaces(),
    ])


def aggressive_pipeline() -> CleaningPipeline:
    """Standard plus punctuation folding. More normalization, more loss."""
    return CleaningPipeline([
        UnicodeNormalize("NFKC"),
        RemoveControlChars(),
        NormalizeWhitespace(),
        NormalizePunctuation(),
        FixHyphenation(),
        CollapseNewlines(unwrap_single=True),
        CollapseSpaces(),
    ])


PIPELINES = {
    "minimal": minimal_pipeline,
    "standard": standard_pipeline,
    "aggressive": aggressive_pipeline,
}


def get_pipeline(name: str) -> CleaningPipeline:
    if name not in PIPELINES:
        raise ValueError(f"Unknown pipeline {name!r}. Options: {list(PIPELINES)}")
    return PIPELINES[name]()