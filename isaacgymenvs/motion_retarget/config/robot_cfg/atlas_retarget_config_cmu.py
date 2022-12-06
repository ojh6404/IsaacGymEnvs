import numpy as np
from pybullet_utils import transformations

ROBOT_URDF_FILENAME = "assets/urdf/atlas/atlas_v4_with_multisense.urdf"

POS_SIZE = 3
ROT_SIZE = 4
DEFAULT_ROT = np.array([0, 0, 0, 1])
FORWARD_DIR = np.array([1,  0, 0])
GROUND_URDF_FILENAME = "plane_implicit.urdf"

ROBOT_ROOT_HEIGHT = 0.092
ROBOT_CHEST_LENGTH = 0.128 + 0.025 + 0.3085   # [m]
ROBOT_HAND_LENGTH = 1.0
ROBOT_LEG_LENGTH = 0.122 + 0.01 + 0.38 + 0.38 + 0.04

REF_POS_SCALE = 5.6
INIT_POS = np.array([0, 0, 0.1])
INIT_ROT = transformations.quaternion_from_euler(
    ai=0, aj=np.pi/2, ak=0, axes="sxyz")
INIT_ROT2 = transformations.quaternion_from_euler(
    ai=0, aj=0, ak=np.pi/2, axes="sxyz")

JOINT_DAMPING = [0.001]
FORWARD_DIR_OFFSET = np.array([0, 0, 0])
SIM_ROOT_OFFSET = np.array([0, 0, -0.06])

# robot's joint : {'back_bkz': 0, 'back_bky': 1, 'back_bkx': 2,
# 'l_arm_shz': 3, 'l_arm_shx': 4, 'l_arm_ely': 5, 'l_arm_elx': 6,
# 'l_arm_wry': 7, 'l_arm_wrx': 8, 'l_arm_wry2': 9, 'neck_ry': 10,
# 'r_arm_shz': 11, 'r_arm_shx': 12, 'r_arm_ely': 13, 'r_arm_elx': 14,
# 'r_arm_wry': 15, 'r_arm_wrx': 16, 'r_arm_wry2': 17, 'l_leg_hpz': 18,
# 'l_leg_hpx': 19, 'l_leg_hpy': 20, 'l_leg_kny': 21, 'l_leg_aky': 22,
# 'l_leg_akx': 23, 'r_leg_hpz': 24, 'r_leg_hpx': 25, 'r_leg_hpy': 26,
# 'r_leg_kny': 27, 'r_leg_aky': 28, 'r_leg_akx': 29}

# scale factor
HIP_TO_SPINE_SCALE = 1.8
SPINE_TO_TORSO_SCALE = 1.8
TORSO_TO_NECK_SCALE = 1.8
NECK_TO_NECK1_SCALE = 1.5
NECK1_TO_HEAD_SCALED = 1.5

TORSO_TO_SHOULDER1_SCALE = 1.8
SHOULDER1_TO_SHOULDER2_SCALE = 2.4
SHOULDER2_TO_ELBOW_SCALE = 1.1  # 1.4
ELBOW_TO_HAND_SCALE = 1.5
HAND_TO_FINGER1_SCALE = 1.5
FINGER1_TO_FINGER2_SCALE = 1.5
FIGNER2_TO_THUMB_SCALE = 1.5

HIPS_TO_CROTCH_SCALE = 1.0
CROTCH_TO_KNEE_SCALE = 1.0
KNEE_TO_FOOT_SCALE = 1.0
FOOT_TO_TOE_SCALE = 1.0
