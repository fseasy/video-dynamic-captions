# -*- coding: utf-8 -*-
"""Types"""
from enum import Enum


class CaptionTextType(str, Enum):
    """How to generate the dynamic caption text"""

    # Random join consequent tokens as a caption.
    # e.g: tokens = ["I", "Love", "Peace", "and", "Nature"], result may be ["I Love", "Peace and", "Nature"] randomly.
    RANDOM_JOIN = "random_join"


class CaptionStyleType(str, Enum):
    """How to select the caption text style when rendering"""

    # Random fit the sytle to the text.
    RANDOM_FIT = "random_fit"


class CaptionOutputType(str, Enum):
    """Caption output type."""

    # subtitles embeded to video softly
    SOFT_EMBED_VIDEO = "soft_embed_video"
    # subtitles embeded to video hardly
    HARD_EMBED_VIDEO = "hard_embed_video"
    # don't process video, just generate the .ass subtitle
    ASS = "ass"
