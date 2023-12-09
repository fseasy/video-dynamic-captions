# -*- coding: utf-8 -*-
import gc
import copy
from pathlib import Path
from typing import List, Optional, Union, Literal

import numpy as np
import whisperx
from whisperx.types import AlignedTranscriptionResult, SingleSegment
from whisperx.audio import SAMPLE_RATE
import torch

from ..conf import CaptionExeConfig, logger
from .types import VideoTimeResult, ScriptTimeUnit, WordTimeUnit


def align2get_word_time(
    audio: Union[str, Path, np.ndarray],
    audio_lang_code: Optional[str],
    script_time_units: Union[List[ScriptTimeUnit], List[SingleSegment]],
    exe_config: CaptionExeConfig,
) -> VideoTimeResult:
    """no Exception will be catch. uppper should process it"""
    if isinstance(audio, (str, Path)):
        audio = whisperx.load_audio(str(audio))
    assert isinstance(audio, np.ndarray)
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
    # script_time_units' type is compatibility to WhisperX's type
    align_result = whisperx.align(script_time_units, align_model, align_meta, audio=audio, device=device)
    filled_script_time_units = _make_script_time_units_from_whipserx(align_result)
    audio_duration_sec = len(audio) / SAMPLE_RATE
    filled_script_time_units = _postprocess_script_time_units(filled_script_time_units, audio_duration_sec)
    final_result: VideoTimeResult = {
        "script_word_times": filled_script_time_units,
        "audio_language_code": audio_lang_code,
    }
    return final_result


def _make_script_time_units_from_whipserx(
    whisperx_aligned_result: AlignedTranscriptionResult,
) -> List[ScriptTimeUnit]:
    # make result initially
    result: List[ScriptTimeUnit] = []
    for single_segment in whisperx_aligned_result["segments"]:
        whisper_words = single_segment["words"]
        word_times: List[WordTimeUnit] = []
        for w in whisper_words:
            wt: WordTimeUnit = {"text": w["word"], "start": w.get("start"), "end": w.get("end")}
            word_times.append(wt)
        script_unit: ScriptTimeUnit = {
            "text": single_segment["text"],
            "start": single_segment["start"],
            "end": single_segment["end"],
            "word_times": word_times,
        }
        result.append(script_unit)
    return result


def _postprocess_script_time_units(
    script_time_units: List[ScriptTimeUnit], audio_duration_sec: float
) -> List[ScriptTimeUnit]:
    def _fill_word_start_end_(result: List[ScriptTimeUnit]):
        """in scripts level, always has start/end time. but"""
        for script_time in result:
            _fill_missing_word_time_by_linear_approximate_(script_time)

    def _limit_end_time_(result: List[ScriptTimeUnit]):
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
                    assert isinstance(w["start"], float) and isinstance(w["end"], float)
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

    def _monitonic(result: List[ScriptTimeUnit]):
        new_result = []
        prev_end = 0.0
        for script_word_time in result:
            if script_word_time["start"] < prev_end:
                logger.warning(
                    "ASR get invalid script: start-time(%s) < prev-end(%s). Drop", script_word_time["start"], prev_end
                )
                continue
            prev_end = script_word_time["end"]
            new_result.append(script_word_time)
        return new_result

    result = copy.deepcopy(script_time_units)
    _fill_word_start_end_(result)
    _limit_end_time_(result)
    result = _monitonic(result)
    return result


def _fill_missing_word_time_by_linear_approximate_(script_time: ScriptTimeUnit):
    """we firstly fill the missing `start` fields, then fill the `end` fields.
    Exception:
        No exception will be catched.
    """
    word_times = script_time["word_times"]
    if not word_times:
        return
    global_start_missing_range_start = 0
    while global_start_missing_range_start < len(word_times):
        # find the next `start`-field-missing range
        rstart = global_start_missing_range_start
        while rstart < len(word_times) and word_times[rstart]["start"] is not None:
            rstart += 1
        if rstart >= len(word_times):
            # no any missing, just exit
            break
        rend = rstart + 1 # range-end for start-field-missing range is not included
        while rend < len(word_times) and word_times[rend]["start"] is None:
            rend += 1
        # Now get the range of missing. we firstly fill the start, then the end
        first_word_prev_char_num = 0
        if rstart == 0:
            # condition1: the first word-time no start, only can use the script-level info
            rstart_sec = script_time["start"]
        else:
            # prev word-time has `start`. Now test whether previous has end
            prev_word_time = word_times[rstart - 1]
            if prev_word_time["end"] is not None:
                # has End. start-sec use this value
                rstart_sec = prev_word_time["end"]
            else:
                # has to use the prev-start. and chars including previous word-time
                assert isinstance(prev_word_time["start"], float)
                rstart_sec = prev_word_time["start"]
                first_word_prev_char_num = len(prev_word_time["text"])
        if rend >= len(word_times):
            # reach the end. test the last word-time whether has `end`
            if word_times[-1]["end"] is not None:
                rend_sec = word_times[-1]["end"]
            else:
                rend_sec = script_time["end"]
        else:
            prev_word_time = word_times[rend - 1]
            if prev_word_time["end"] is not None:
                rend_sec = prev_word_time["end"]
            else:
                _ws = word_times[rend]["start"]
                assert _ws is not None
                rend_sec = _ws
        range_sec = rend_sec - rstart_sec
        range_char_num = first_word_prev_char_num
        for idx in range(rstart, rend):
            range_char_num += len(word_times[idx]["text"])
        sec_per_char = range_sec / (range_char_num + 1e-6)
        prev_start_sec = rstart_sec
        prev_char_num = first_word_prev_char_num
        for idx in range(rstart, rend):
            approx_start_sec = prev_start_sec + prev_char_num * sec_per_char
            word_times[idx]["start"] = approx_start_sec
            prev_start_sec = approx_start_sec
            prev_char_num = len(word_times[idx]["text"])
        global_start_missing_range_start = rend
    for idx in range(0, len(word_times) - 1):
        if word_times[idx]["end"] is None:
            word_times[idx]["end"] = word_times[idx + 1]["start"]
    if word_times[-1]["end"] is None:
        word_times[-1]["end"] = script_time["end"]


