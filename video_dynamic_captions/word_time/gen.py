# -*- coding: utf-8 -*-
"""Make Captions."""

from pathlib import Path
from typing import Optional, Union

import pysubs2

from ..conf import CaptionExeConfig
from ..exceptions import ParamError
from .types import VideoWordTimeResult
from .gen_from_subtitle import gen_from_subtitle
from .gen_from_asr import gen_from_asr


class WordTimeGen(object):
    def __init__(
        self,
        audio_path: Union[str, Path],
        audio_lang_code: Optional[str] = None,
        subtitle_path: Optional[Union[str, Path]] = None,
        exe_config: Optional[CaptionExeConfig] = None,
    ):
        self._audio_path = audio_path
        self._audio_lang_code = audio_lang_code
        self._subtitle_path = subtitle_path
        self._exe_config = exe_config if exe_config is not None else CaptionExeConfig()

    def __call__(self) -> VideoWordTimeResult:
        """
        Exceptions
        =====
        SubtitleParseError
        ASRAlignWordTimeError
        ASRTranscribeError
        """
        if self._subtitle_path:
            return gen_from_subtitle(
                audio_path=self._audio_path,
                audio_lang_code=self._audio_lang_code,
                subtitle_path=self._subtitle_path,
                exe_config=self._exe_config,
            )
        return gen_from_asr(
            audio_path=self._audio_path,
            audio_lang_code=self._audio_lang_code,
            exe_config=self._exe_config,
        )
