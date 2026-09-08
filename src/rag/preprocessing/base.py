"""
Cleaning step interface.

Each cleaning operation is a separate, independently toggleable step rather
than one monolithic clean() function. This matters because "how aggressively
should text be cleaned?" is an EXPERIMENTAL QUESTION, not a settled one.
In Phase 15 we compare pipelines by changing configuration, not code.
"""

from abc import ABC, abstractmethod


class CleaningStep(ABC):
    """A single text transformation."""

    @abstractmethod
    def apply(self, text: str) -> str:
        """Transform the text. Must be safe to call on an empty string."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}()"


class CleaningPipeline:
    """Applies an ordered sequence of cleaning steps.

    Order matters. Unicode normalization must run before regex-based steps,
    because the regexes assume normalized input — a pattern matching '-'
    will not match the Unicode minus sign U+2212 until normalization has
    converted it.
    """

    def __init__(self, steps: list[CleaningStep]) -> None:
        self.steps = steps

    def clean(self, text: str) -> str:
        for step in self.steps:
            text = step.apply(text)
        return text

    def clean_verbose(self, text: str) -> tuple[str, list[tuple[str, int, int]]]:
        """Clean while recording each step's effect on length.

        Returns the cleaned text plus a list of (step_name, chars_before,
        chars_after). Used for inspecting what each step actually does —
        a step that removes 40% of your corpus deserves scrutiny.
        """
        history: list[tuple[str, int, int]] = []
        for step in self.steps:
            before = len(text)
            text = step.apply(text)
            history.append((step.name, before, len(text)))
        return text, history

    def __repr__(self) -> str:
        names = " -> ".join(s.name for s in self.steps)
        return f"CleaningPipeline({names})"