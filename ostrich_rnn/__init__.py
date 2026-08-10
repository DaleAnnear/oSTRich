"""RNN-based short tandem repeat detection package."""

from .motifs import MotifVocab

__all__ = ["MotifVocab", "RNNSTRDetector"]


def __getattr__(name: str):
    if name == "RNNSTRDetector":
        from .model import RNNSTRDetector

        return RNNSTRDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
