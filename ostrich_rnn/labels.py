"""Shared label constants."""

from enum import IntEnum


class RepeatState(IntEnum):
    """Per-base repeat state labels."""

    OUTSIDE = 0
    START = 1
    INSIDE = 2
    END = 3


PAD_STATE_LABEL = -100
PAD_MOTIF_LABEL = -100
