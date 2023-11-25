# -*- coding: utf-8 -*-
"""Config defination"""
import logging
from dataclasses import dataclass
from typing import Tuple

import GPUtil

LOG_NAME = "vdcaption"

logger = logging.getLogger(LOG_NAME)

@dataclass
class CaptionExeConfig:
    log_level: str = "INFO"
    # ASR(WhisperX) config
    # see https://github.com/m-bain/whisperX#usage--command-line for available setting
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 16

    def get_valid_device_compute_type(self) -> Tuple[str, str]:
        def _auto(cuda_compute_type):
            if GPUtil.getGPUs():
                return ("cuda", cuda_compute_type)
            # no gpu. use default = CPU+int8; If your cpu not support int8, set to float32 instead
            return ("cpu", "int8")

        if self.device == "cpu":
            return (self.device, self.compute_type)
        elif self.device == "cuda":
            return _auto(self.compute_type)
        else:
            # invalid
            return _auto("float16")
