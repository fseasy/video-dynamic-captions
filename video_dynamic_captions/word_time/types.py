# -*- coding: utf-8 -*-
from typing import TypedDict, List

class WordTimeUnit(TypedDict):
    text: str
    start: float # in seconds
    end: float # in seconds


class ScriptWordTimeUnit(TypedDict):
    text: str # script level text
    start: float # in seconds
    end: float # in seconds
    word_times: List[WordTimeUnit]

class VideoWordTimeResult(TypedDict):
    script_word_times: List[ScriptWordTimeUnit]
    audio_language_code: str