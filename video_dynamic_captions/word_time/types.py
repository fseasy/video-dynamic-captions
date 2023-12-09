# -*- coding: utf-8 -*-
from typing import TypedDict, List, Optional


class WordTimeUnit(TypedDict):
    text: str
    # Align Model may give empty result(None). but we will approximate it to get a valid float value.
    start: Optional[float]  # in seconds. 
    end: Optional[float]  # in seconds


class ScriptTimeUnit(TypedDict):
    text: str  # script level text
    start: float  # in seconds
    end: float  # in seconds
    word_times: List[WordTimeUnit]


class VideoTimeResult(TypedDict):
    script_word_times: List[ScriptTimeUnit]
    audio_language_code: str
