# -*- coding: utf-8 -*-
import gc
import copy
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import whisperx
from whisperx.types import AlignedTranscriptionResult, SingleSegment
from whisperx.audio import SAMPLE_RATE
import torch

from ..conf import CaptionExeConfig, logger
from .types import VideoWordTimeResult, ScriptWordTimeUnit, WordTimeUnit


def align2get_word_time(
    audio: Union[str, Path, np.ndarray],
    audio_lang_code: Optional[str],
    script_word_time_units: Union[List[ScriptWordTimeUnit], List[SingleSegment]],
    exe_config: CaptionExeConfig,
) -> VideoWordTimeResult:
    """no Exception will be catch. uppper should process it"""
    if isinstance(audio, (str, Path)):
        audio = whisperx.load_audio(str(audio))
    device, compute_type = exe_config.get_valid_device_compute_type()
    if not audio_lang_code:
        logger.info("No audio-language code given, auto-detect by whipser")
        whisper = whisperx.load_model(exe_config.model, device=device, compute_type=compute_type)
        audio_lang_code = whisper.detect_language(audio)
        assert audio_lang_code, "Audio Language Code still None after auto-detect"
        whisper = None
        torch.cuda.empty_cache()
        gc.collect()

    _EXTRA_CODE2MODEL_NAME = {
        "ro": "anton-l/wav2vec2-large-xlsr-53-romanian",
        "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    }
    align_model_name = _EXTRA_CODE2MODEL_NAME.get(audio_lang_code, None)

    align_model, align_meta = whisperx.load_align_model(audio_lang_code, device=device, model_name=align_model_name)
    # script_word_time_units' type is compatibility to WhisperX's type
    align_result = whisperx.align(script_word_time_units, align_model, align_meta, audio=audio, device=device)
    filled_script_word_time_units = _make_script_word_time_units_from_whipserx(align_result)
    audio_duration_sec = len(audio) / SAMPLE_RATE
    filled_script_word_time_units = _postprocess_script_word_time_units(
        filled_script_word_time_units, audio_duration_sec
    )
    final_result: VideoWordTimeResult = {
        "script_word_times": filled_script_word_time_units,
        "audio_language_code": audio_lang_code
    }
    return final_result


def _make_script_word_time_units_from_whipserx(
    whisperx_aligned_result: AlignedTranscriptionResult,
) -> List[ScriptWordTimeUnit]:
    # make result initially
    result: List[ScriptWordTimeUnit] = []
    for single_segment in whisperx_aligned_result["segments"]:
        whisper_words = single_segment["words"]
        word_times: List[WordTimeUnit] = []
        for w in whisper_words:
            wt: WordTimeUnit = {"text": w["word"], "start": w.get("start"), "end": w.get("end")}
            word_times.append(wt)
        script_unit: ScriptWordTimeUnit = {
            "text": single_segment["text"],
            "start": single_segment["start"],
            "end": single_segment["end"],
            "word_times": word_times,
        }
        result.append(script_unit)
    return result


def _postprocess_script_word_time_units(
    script_word_time_units: List[ScriptWordTimeUnit], audio_duration_sec: float
) -> List[ScriptWordTimeUnit]:
    def _fill_word_start_end_(result: List[ScriptWordTimeUnit]):
        """in scripts level, always has start/end time. but"""
        for wts in result:
            for idx, w in enumerate(wts["word_times"]):
                if w["start"] is None:
                    w["start"] = _get_linear_approximate_timecode(wts, idx, "start")
                if w["end"] is None:
                    w["end"] = _get_linear_approximate_timecode(wts, idx, "end")

    def _limit_end_time_(result: List[ScriptWordTimeUnit]):
        processing_idx = len(result) - 1
        while processing_idx >= 0:
            script_wt = result[processing_idx]
            processing_idx -= 1
            start = script_wt["start"]
            if start >= audio_duration_sec:
                logger.warning(
                    "ASR get invalid script: start-time(%s) >= audio_duration(%s). Drop", start, audio_duration_sec
                )
                result.pop()
                continue
            end = script_wt["end"]
            if end > audio_duration_sec:
                script_wt["end"] = audio_duration_sec
                # word-level
                drop_words = []
                word_times = script_wt["word_times"]
                widx = len(word_times) - 1
                while widx >= 0:
                    w = word_times[widx]
                    widx -= 1
                    if w["start"] >= audio_duration_sec:
                        drop_words.append(w)
                        word_times.pop()
                        continue
                    if w["end"] >= audio_duration_sec:
                        w["end"] = audio_duration_sec
                    break
                logger.warning(
                    "ASR get invalid script: end-time(%s) > audio_duration(%s). Reset end and drop words(%s)",
                    end,
                    audio_duration_sec,
                    drop_words,
                )
            # finish all
            break

    def _monitonic(result: List[ScriptWordTimeUnit]):
        new_result = []
        prev_end = 0
        for script_word_time in result:
            if script_word_time["start"] < prev_end:
                logger.warning(
                    "ASR get invalid script: start-time(%s) < prev-end(%s). Drop", script_word_time["start"], prev_end
                )
                continue
            prev_end = script_word_time["end"]
            new_result.append(script_word_time)
        return new_result

    result = copy.deepcopy(script_word_time_units)
    _fill_word_start_end_(result)
    _limit_end_time_(result)
    result = _monitonic(result)
    return result


def _get_linear_approximate_timecode(
    script_word_time_unit: ScriptWordTimeUnit, word_idx: int, time_code_name: str
) -> float:
    assert time_code_name in set(["start", "end"])
    word_times = script_word_time_unit["word_times"]
    valid_start_idx = word_idx - 1
    while valid_start_idx >= 0:
        wt = word_times[valid_start_idx]
        if wt[time_code_name] is not None:
            break
        valid_start_idx -= 1
    range_start_time = (
        word_times[valid_start_idx][time_code_name] if valid_start_idx >= 0 else script_word_time_unit["start"]
    )

    valid_end_idx = word_idx + 1
    while valid_end_idx < len(word_times):
        wt = word_times[valid_end_idx]
        if wt[time_code_name] is not None:
            break
        valid_end_idx += 1
    range_end_time = (
        word_times[valid_end_idx][time_code_name] if valid_end_idx < len(word_times) else script_word_time_unit["end"]
    )
    range_interval = range_end_time - range_start_time
    assert range_interval >= 0, (
        "Linear Approximate get invalid interval: "
        f"{range_start_time}=>{range_end_time}(interval={range_interval} < 0)"
    )
    percent = (word_idx - valid_start_idx) / (valid_end_idx - valid_start_idx)
    return range_start_time + percent * range_interval
