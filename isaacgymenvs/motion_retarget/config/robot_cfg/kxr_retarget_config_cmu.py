"""
This file should be adjusted by different target robot with different scale size and structure
"""
import numpy as np
from pybullet_utils import transformations

ROBOT_URDF_FILENAME = "assets/urdf/kxr/kxrl2l5a3g/urdf/kxrl2l5a3.urdf"

POS_SIZE = 3
ROT_SIZE = 4
DEFAULT_ROT = np.array([0, 0, 0, 1])
FORWARD_DIR = np.array([1,  0, 0])
OUTPUT_FILENAME = "kxrl2l5a3g"
GROUND_URDF_FILENAME = "plane_implicit.urdf"

KXR_ROOT_HEIGHT = 0.092
# KXR_CHEST_LENGTH = 0.128 + 0.025 + 0.3085   # [m]
# KXR_HAND_LENGTH = 1.0
# KXR_LEG_LENGTH = 0.122 + 0.01 + 0.38 + 0.38 + 0.04

# REF_POS_SCALE = 0.68
REF_POS_SCALE = 0.95
INIT_POS = np.array([0, 0, 0.15])
INIT_ROT = transformations.quaternion_from_euler(
    ai=0, aj=np.pi/2, ak=0, axes="sxyz")
INIT_ROT2 = transformations.quaternion_from_euler(
    ai=0, aj=0, ak=np.pi/2, axes="sxyz")

Z_OFFSET = 0.00

JOINT_DAMPING = [0.001]
FORWARD_DIR_OFFSET = np.array([0, 0, 0])
SIM_ROOT_OFFSET = np.array([0, 0, -0.06])

# scale factor for 77
HIP_TO_SPINE_SCALE = 2.5
SPINE_TO_TORSO_SCALE = 2.0
TORSO_TO_NECK_SCALE = 2.0
NECK_TO_NECK1_SCALE = 2.0
NECK1_TO_HEAD_SCALED = 1.5

TORSO_TO_SHOULDER1_SCALE = 1.5  # 1.8
SHOULDER1_TO_SHOULDER2_SCALE = 2.0  # 1.8
SHOULDER2_TO_ELBOW_SCALE = 1.0  # 1.4
ELBOW_TO_HAND_SCALE = 2.0
HAND_TO_FINGER1_SCALE = 2.0
FINGER1_TO_FINGER2_SCALE = 1.5
FIGNER2_TO_THUMB_SCALE = 1.5

HIPS_TO_CROTCH_SCALE = 3.5  # 0.7
CROTCH_TO_KNEE_SCALE = 0.0  # 0.8
KNEE_TO_FOOT_SCALE = 1.2  # 1.0
FOOT_TO_TOE_SCALE = 1.0
