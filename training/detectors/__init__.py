import os
import sys
import logging
from importlib import import_module
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR

logger = logging.getLogger(__name__)


def _optional_import(module_name, class_name):
    try:
        module = import_module(f".{module_name}", __name__)
        return getattr(module, class_name)
    except Exception as exc:
        logger.warning("Skip optional detector import %s.%s: %s", module_name, class_name, exc)
        return None


_optional_import("utils.slowfast", "setup_environment")


from .facexray_detector import FaceXrayDetector
from .xception_detector import XceptionDetector
from .efficientnetb4_detector import EfficientDetector
from .resnet34_detector import ResnetDetector
from .meso4_detector import Meso4Detector
from .meso4Inception_detector import Meso4InceptionDetector

_optional_import("f3net_detector", "F3netDetector")
_optional_import("spsl_detector", "SpslDetector")
_optional_import("core_detector", "CoreDetector")
_optional_import("capsule_net_detector", "CapsuleNetDetector")
_optional_import("srm_detector", "SRMDetector")
_optional_import("ucf_detector", "UCFDetector")
_optional_import("recce_detector", "RecceDetector")
_optional_import("fwa_detector", "FWADetector")
_optional_import("ffd_detector", "FFDDetector")
_optional_import("videomae_detector", "VideoMAEDetector")
_optional_import("clip_detector", "CLIPDetector")
_optional_import("timesformer_detector", "TimeSformerDetector")
_optional_import("xclip_detector", "XCLIPDetector")
_optional_import("sbi_detector", "SBIDetector")
_optional_import("ftcn_detector", "FTCNDetector")
_optional_import("i3d_detector", "I3DDetector")
_optional_import("altfreezing_detector", "AltFreezingDetector")
_optional_import("stil_detector", "STILDetector")
_optional_import("lsda_detector", "LSDADetector")
_optional_import("sladd_detector", "SLADDXceptionDetector")
_optional_import("pcl_xception_detector", "PCLXceptionDetector")
_optional_import("iid_detector", "IIDDetector")
_optional_import("lrl_detector", "LRLDetector")
_optional_import("rfm_detector", "RFMDetector")
_optional_import("uia_vit_detector", "UIAViTDetector")
_optional_import("multi_attention_detector", "MultiAttentionDetector")
_optional_import("sia_detector", "SIADetector")
_optional_import("tall_detector", "TALLDetector")
_optional_import("effort_detector", "EffortDetector")
