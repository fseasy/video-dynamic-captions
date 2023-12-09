# -*- coding: utf-8 -*-
"""Generate word-time from ASR"""

import gc
from pathlib import Path
from typing import List, Optional, Union

import whisperx

from ..conf import CaptionExeConfig, logger
from .types import ScriptTimeUnit, VideoTimeResult, WordTimeUnit
from .align import align2get_word_time
from ..exceptions import ASRTranscribeError, ASRAlignWordTimeError


def gen_from_asr(
    audio_path: Union[str, Path], audio_lang_code: Optional[str], exe_config: CaptionExeConfig
) -> VideoTimeResult:
    device, compute_type = exe_config.get_valid_device_compute_type()
    try:
        audio = whisperx.load_audio(str(audio_path))
        whisper_model = whisperx.load_model(exe_config.model, device=device, compute_type=compute_type)
        transcribe_result = whisper_model.transcribe(audio, exe_config.batch_size, language=audio_lang_code)
        audio_lang_code = transcribe_result["language"]
    except Exception as e:
        raise ASRTranscribeError(e) from e
    try:
        return align2get_word_time(
            audio, audio_lang_code, script_time_units=transcribe_result["segments"], exe_config=exe_config
        )
    except Exception as e:
        raise ASRAlignWordTimeError(e) from e