from .detector_template import DetectorTemplate
from .key_point_pose import KeyPoint2Pose
from .center_det import CenterDet
from .center_det_laam6d import CenterDetLaam6d
from .center_det_kd import CenterDetKD
from .center_det_mt import CenterDetMT
__all__ = {
    'DetectorTemplate': DetectorTemplate,
    'KeyPoint2Pose': KeyPoint2Pose,
    'CenterDet': CenterDet,
    'CenterDetLaam6d':CenterDetLaam6d,
    'CenterDetKD': CenterDetKD,
    'CenterDetMT': CenterDetMT,
}


def build_detector(model_cfg, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, dataset=dataset
    )
    return model
