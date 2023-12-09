# -*- coding: utf-8 -*-
"""Test Word-Time"""

from pathlib import Path
from typing import Literal

import pytest

from video_dynamic_captions.word_time import gen_from_subtitle
from video_dynamic_captions.word_time import align
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
            id="surrounding-has",
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
                    WordTimeUnit(text="1", start=0., end=2.),
                    WordTimeUnit(text="2", start=2., end=4.),
                    WordTimeUnit(text="3", start=4., end=6.),
                ],
            ),
            id="only-script-level-has",
        ),
    ],
)
def test_fill_missing_word_time_by_linear_approximate_(
    script_time: ScriptTimeUnit, expect_value: ScriptTimeUnit
):
    align._fill_missing_word_time_by_linear_approximate_(script_time)
    for idx, filled_wt in enumerate(script_time["word_times"]):
        expect_wt = expect_value["word_times"][idx]
        assert filled_wt["start"] == pytest.approx(expect_wt["start"], 1e-2)
        assert filled_wt["end"] == pytest.approx(expect_wt["end"], 1e-2)
