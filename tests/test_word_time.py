# -*- coding: utf-8 -*-
"""Test Word-Time"""

from pathlib import Path

import pytest

from video_dynamic_captions.word_time import gen_from_subtitle


def test_subtitle_parse(sample_srt_path, target_scripts):
    script_wts = gen_from_subtitle._parse_subtitle(sample_srt_path)
    assert len(script_wts) == len(target_scripts)
    for script_wt, target_wt in zip(script_wts, target_scripts):
        assert script_wt["text"] == target_wt["text"]
        assert abs(script_wt["start"] - target_wt["start"]) < 1e-6
        assert abs(script_wt["end"] - target_wt["end"]) < 1e-6