import os
import sys
import logging
from importlib import import_module
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import LOSSFUNC

logger = logging.getLogger(__name__)


def _optional_import(module_name, class_name):
    try:
        module = import_module(f".{module_name}", __name__)
        return getattr(module, class_name)
    except Exception as exc:
        logger.warning("Skip optional loss import %s.%s: %s", module_name, class_name, exc)
        return None


from .cross_entropy_loss import CrossEntropyLoss
from .consistency_loss import ConsistencyCos
from .capsule_loss import CapsuleLoss
from .bce_loss import BCELoss
from .am_softmax import AMSoftmaxLoss
from .am_softmax import AMSoftmax_OHEM
from .contrastive_regularization import ContrastiveLoss
from .l1_loss import L1Loss
from .id_loss import IDLoss
from .js_loss import JS_Loss
from .patch_consistency_loss import PatchConsistencyLoss
from .supercontrast_loss import SupConLoss
_optional_import("vgg_loss", "VGGLoss")
_optional_import("region_independent_loss", "RegionIndependentLoss")
