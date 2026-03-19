import os
import sys
import logging
from importlib import import_module
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import BACKBONE

logger = logging.getLogger(__name__)


def _optional_import(module_name, class_name):
    try:
        module = import_module(f".{module_name}", __name__)
        return getattr(module, class_name)
    except Exception as exc:
        logger.warning("Skip optional backbone import %s.%s: %s", module_name, class_name, exc)
        return None


from .xception import Xception
from .mesonet import Meso4, MesoInception4
from .resnet34 import ResNet34
from .xception_sladd import Xception_SLADD
_optional_import("efficientnetb4", "EfficientNetB4")
