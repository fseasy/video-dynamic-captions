# -*- coding: utf-8 -*-
"""Exceptions"""


class VDCBaseError(Exception):
    """Base Error.
    You could use it to capture all exceptions from this program"""


class ParamError(VDCBaseError):
    """Input Param Error. It means your input has error, not our program exception"""


class SubtitleParseError(VDCBaseError):
    """Parse given Subtitle Failed"""


class ASRAlignWordTimeError(VDCBaseError):
    """Failed when align word time using ASR"""


class ASRTranscribeError(VDCBaseError):
    """Transcribe audio error"""
