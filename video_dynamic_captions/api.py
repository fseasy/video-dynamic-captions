# -*- coding: utf-8 -*-
"""Video Dynamic Captions interface"""
from pathlib import Path
from typing import Union, Optional

from .types import CaptionTextType, CaptionStyleType, CaptionOutputType
from .conf import CaptionExeConfig

def caption(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    subtitle_path: Optional[Union[str, Path]] = None,
    caption_text_type: CaptionTextType = CaptionTextType.RANDOM_JOIN,
    caption_style_type: CaptionStyleType = CaptionStyleType.RANDOM_FIT,
    output_type: CaptionOutputType = CaptionOutputType.SOFT_EMBED_VIDEO,
    exe_config: Optional[CaptionExeConfig] = None
):
    ...