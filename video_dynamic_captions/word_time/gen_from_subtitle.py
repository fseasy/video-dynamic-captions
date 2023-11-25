# -*- coding: utf-8 -*-
"""Generate from subtitle"""
from pathlib import Path
from typing import List, Optional, Union

import pysubs2

from ..conf import CaptionExeConfig, logger
from .types import ScriptWordTimeUnit, VideoWordTimeResult
from .align import align2get_word_time
from ..exceptions import SubtitleParseError, ASRAlignWordTimeError

def gen_from_subtitle(
    audio_path: Union[str, Path],
    audio_lang_code: Optional[str],
    subtitle_path: Union[str, Path],
    exe_config: CaptionExeConfig,
) -> VideoWordTimeResult:
    try:
        script_units = _parse_subtitle(subtitle_path)
    except Exception as e:
        raise SubtitleParseError(e) from e
    try:
        return align2get_word_time(
            audio=audio_path,
            audio_lang_code=audio_lang_code,
            script_word_time_units=script_units,
            exe_config=exe_config,
        )
    except Exception as e:
        raise ASRAlignWordTimeError(e) from e


def _parse_subtitle(subtitle_path: Union[str, Path]) -> List[ScriptWordTimeUnit]:
    """Don't process exceptions => let upper process"""
    with open(subtitle_path, encoding="utf-8", errors="replace") as f:
        sub = pysubs2.SSAFile.from_file(f)
    script_units: List[ScriptWordTimeUnit] = []
    for event in sub.events:
        if event.is_comment:
            continue
        start_sec = event.start / 1_000
        end_sec = event.end / 1_000
        text = event.plaintext
        unit: ScriptWordTimeUnit = {
            "text": text,
            "start": start_sec,
            "end": end_sec,
            "word_times": [],  # leave it to fill in future steps
        }
        script_units.append(unit)
    return script_units
