import os
import sys
import logging
from importlib import import_module
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

logger = logging.getLogger(__name__)


def _optional_import(module_name, class_name):
    try:
        module = import_module(f".{module_name}", __name__)
        return getattr(module, class_name)
    except Exception as exc:
        logger.warning("Skip optional dataset import %s.%s: %s", module_name, class_name, exc)
        return None


from .abstract_dataset import DeepfakeAbstractBaseDataset
_optional_import("I2G_dataset", "I2GDataset")
_optional_import("iid_dataset", "IIDDataset")
_optional_import("ff_blend", "FFBlendDataset")
_optional_import("fwa_blend", "FWABlendDataset")
_optional_import("lrl_dataset", "LRLDataset")
_optional_import("pair_dataset", "pairDataset")
_optional_import("sbi_dataset", "SBIDataset")
_optional_import("lsda_dataset", "LSDADataset")
_optional_import("tall_dataset", "TALLDataset")
