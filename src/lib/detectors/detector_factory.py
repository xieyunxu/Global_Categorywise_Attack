from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
# Ctdet
#from .ctdet import CtdetDetector
from .ctdet_uni import CtdetDetector
# Multi_pose
#from .multi_pose import MultiPoseDetector
from .multipose_uni import MultiPoseDetector


detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'multi_pose': MultiPoseDetector, 
}
