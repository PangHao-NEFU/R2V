# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# backwards compatibility to support `from OFA_OCR.fairseq.X import Y`
from OFA_OCR.fairseq.distributed import utils as distributed_utils
from OFA_OCR.fairseq.logging import meters, metrics, progress_bar  # noqa

sys.modules["OFA_OCR.fairseq.distributed_utils"] = distributed_utils
sys.modules["OFA_OCR.fairseq.meters"] = meters
sys.modules["OFA_OCR.fairseq.metrics"] = metrics
sys.modules["OFA_OCR.fairseq.progress_bar"] = progress_bar

# initialize hydra
from OFA_OCR.fairseq.dataclass.initialize import hydra_init
hydra_init()

import OFA_OCR.fairseq.criterions  # noqa
import OFA_OCR.fairseq.distributed  # noqa
import OFA_OCR.fairseq.models  # noqa
import OFA_OCR.fairseq.modules  # noqa
import OFA_OCR.fairseq.optim  # noqa
import OFA_OCR.fairseq.optim.lr_scheduler  # noqa
import OFA_OCR.fairseq.pdb  # noqa
import OFA_OCR.fairseq.scoring  # noqa
import OFA_OCR.fairseq.tasks  # noqa
import OFA_OCR.fairseq.token_generation_constraints  # noqa

import OFA_OCR.fairseq.benchmark  # noqa
import OFA_OCR.fairseq.model_parallel  # noqa
