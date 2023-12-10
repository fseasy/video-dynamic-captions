# -*- coding: utf-8 -*-
"""Test Word-Time"""

from typing import List

import pytest

from video_dynamic_captions.word_time import align
from video_dynamic_captions.word_time import gen_from_subtitle
from video_dynamic_captions.word_time.types import ScriptTimeUnit, WordTimeUnit


## gen-from-subtitle


def test_subtitle_parse(sample_srt_path, target_scripts):
    script_wts = gen_from_subtitle._parse_subtitle(sample_srt_path)
    assert len(script_wts) == len(target_scripts)
    for script_wt, target_wt in zip(script_wts, target_scripts):
        assert script_wt["text"] == target_wt["text"]
        assert abs(script_wt["start"] - target_wt["start"]) < 1e-6
        assert abs(script_wt["end"] - target_wt["end"]) < 1e-6


## align


@pytest.mark.parametrize(
    "script_time, expect_value",
    [
        pytest.param(
            ScriptTimeUnit(
                text="123",
                start=0.0,
                end=10.0,
                word_times=[
                    WordTimeUnit(text="1", start=0.0, end=2.0),
                    WordTimeUnit(text="2", start=None, end=None),
                    WordTimeUnit(text="3", start=8.0, end=10.0),
                ],
            ),
            ScriptTimeUnit(
                text="123",
                start=0.0,
                end=10.0,
                word_times=[
                    WordTimeUnit(text="1", start=0.0, end=2.0),
                    WordTimeUnit(text="2", start=2.0, end=8.0),
                    WordTimeUnit(text="3", start=8.0, end=10.0),
                ],
            ),
            id="surrounding-exist",
        ),
        pytest.param(
            ScriptTimeUnit(
                text="123",
                start=0.0,
                end=6.0,
                word_times=[
                    WordTimeUnit(text="1", start=None, end=None),
                    WordTimeUnit(text="2", start=None, end=None),
                    WordTimeUnit(text="3", start=None, end=None),
                ],
            ),
            ScriptTimeUnit(
                text="123",
                start=0.0,
                end=6.0,
                word_times=[
                    WordTimeUnit(text="1", start=0.0, end=2.0),
                    WordTimeUnit(text="2", start=2.0, end=4.0),
                    WordTimeUnit(text="3", start=4.0, end=6.0),
                ],
            ),
            id="only-script-level-exist",
        ),
        pytest.param(
            ScriptTimeUnit(
                text="12345",
                start=0.0,
                end=8.0,
                word_times=[
                    WordTimeUnit(text="1", start=1.0, end=None),
                    WordTimeUnit(text="23", start=None, end=None),
                    WordTimeUnit(text="4", start=None, end=5.0),
                    WordTimeUnit(text="5", start=6.0, end=8.0),
                ],
            ),
            ScriptTimeUnit(
                text="1234",
                start=0.0,
                end=8.0,
                word_times=[
                    WordTimeUnit(text="1", start=1.0, end=2.0),
                    WordTimeUnit(text="23", start=2.0, end=4.0),
                    WordTimeUnit(text="4", start=4.0, end=5.0),
                    WordTimeUnit(text="5", start=6.0, end=8.0),
                ],
            ),
            id="surrounding-start-block-end-miss-end-block-end-exist",
        ),
    ],
)
def test_fill_missing_word_time_by_linear_approximate_(script_time: ScriptTimeUnit, expect_value: ScriptTimeUnit):
    align._fill_missing_word_time_by_linear_approximate_(script_time)
    for idx, filled_wt in enumerate(script_time["word_times"]):
        expect_wt = expect_value["word_times"][idx]
        assert filled_wt["start"] == pytest.approx(expect_wt["start"], 1e-2)
        assert filled_wt["end"] == pytest.approx(expect_wt["end"], 1e-2)


@pytest.mark.parametrize(
    "script_time_units, audio_duration_sec, expect_script_time_units",
    [
        pytest.param(
            [
                ScriptTimeUnit(
                    text="",
                    start=0.0,
                    end=11.0,
                    word_times=[WordTimeUnit(text="", start=0, end=5.0), WordTimeUnit(text="", start=5.2, end=11.0)],
                ),
                ScriptTimeUnit(
                    text="",
                    start=11.0,
                    end=13.0,
                    word_times=[WordTimeUnit(text="", start=11.0, end=13.0)],
                ),
            ],
            10.0,
            [
                ScriptTimeUnit(
                    text="",
                    start=0.0,
                    end=10.0,
                    word_times=[WordTimeUnit(text="", start=0, end=5.0), WordTimeUnit(text="", start=5.2, end=10.0)],
                ),
            ],
        )
    ],
)
def test_postprocess_script_time_units(
    script_time_units: List[ScriptTimeUnit], audio_duration_sec: float, expect_script_time_units: List[ScriptTimeUnit]
):
    result = align._postprocess_script_time_units(script_time_units, audio_duration_sec)
    assert len(result) == len(expect_script_time_units)
    for idx, processed_t in enumerate(result):
        expect_t = expect_script_time_units[idx]
        assert processed_t == expect_t
