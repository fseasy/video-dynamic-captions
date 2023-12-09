# -*- coding: utf-8 -*-
"""Test data define"""
from pathlib import Path
from typing import List

import pytest

from video_dynamic_captions.word_time.types import ScriptWordTimeUnit


@pytest.fixture(scope="session")
def test_data_dir():
    path = Path(__file__).parent / "data"
    return path


@pytest.fixture(scope="session")
def sample_video_path(test_data_dir: Path):
    return test_data_dir / "sample.mp4"


@pytest.fixture(scope="session")
def sample_audio_path(test_data_dir: Path):
    return test_data_dir / "sample.mp3"


@pytest.fixture(scope="session", autouse=True)
def sample_srt_path(test_data_dir: Path):
    return test_data_dir / "sample.srt"


@pytest.fixture(scope="session", autouse=True)
def target_scripts() -> List[ScriptWordTimeUnit]:
    script_wts: List[ScriptWordTimeUnit] = [
        {
            "text": "Together, we will make America strong again.",
            "start": 0.506,
            "end": 4.309,
            "word_times": [],
        },
        {
            "text": "We will make America wealthy again.",
            "start": 4.989,
            "end": 7.711,
            "word_times": [],
        },
        {
            "text": "We will make America proud again.",
            "start": 8.131,
            "end": 10.833,
            "word_times": [],
        },
        {
            "text": "We will make America safe again.",
            "start": 10.853,
            "end": 13.935,
            "word_times": [],
        },
        {
            "text": "And yes, together, we will make America great again.",
            "start": 14.395,
            "end": 20.019,
            "word_times": [],
        },
    ]
    return script_wts
