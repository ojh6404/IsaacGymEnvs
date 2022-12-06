"""
This file should be adjusted by different target robot with different scale size and structure
"""
import numpy as np
from pybullet_utils import transformations

ROBOT_URDF_FILENAME = "assets/urdf/khr/khr.urdf"

POS_SIZE = 3
ROT_SIZE = 4
DEFAULT_ROT = np.array([0, 0, 0, 1])
FORWARD_DIR = np.array([1,  0, 0])
OUTPUT_FILENAME = "../output/retarget_out_file/khr_motion.txt"
GROUND_URDF_FILENAME = "plane_implicit.urdf"

KHR_ROOT_HEIGHT = 0.092
KHR_CHEST_LENGTH = 0.128 + 0.025 + 0.3085   # [m]
KHR_HAND_LENGTH = 1.0
KHR_LEG_LENGTH = 0.122 + 0.01 + 0.38 + 0.38 + 0.04

# REF_POS_SCALE = 0.68
REF_POS_SCALE = 0.7
INIT_POS = np.array([0, 0, 0.1])
INIT_ROT = transformations.quaternion_from_euler(
    ai=0, aj=np.pi/2, ak=0, axes="sxyz")
INIT_ROT2 = transformations.quaternion_from_euler(
    ai=0, aj=0, ak=np.pi/2, axes="sxyz")

# [leftankle, rightankle, lefthand, righthand]
SIM_TOE_JOINT_IDS = [14, 9, 5, 2]
# [lefthip,righthip,leftshoulder,rightshoulder]
SIM_HIP_JOINT_IDS = [11, 6, 3, 0]
SIM_ROOT_OFFSET = np.array([0, 0, -0.06])

SIM_TOE_OFFSET_LOCAL = [
    np.array([-0.0, 0.02, -0.0]),  # leftleg
    np.array([-0.0, -0.02, 0.0]),  # rightleg
    np.array([-0.0, -0.0, -0.0]),  # lefthand
    np.array([-0.0, 0.0, 0.0])     # righthand
]

JOINT_DAMPING = [0.001]*17
FORWARD_DIR_OFFSET = np.array([0, 0, 0])
SIM_ROOT_OFFSET = np.array([0, 0, -0.06])
# JOINT_DAMPING = [0.1, -0.001, 0.1, 0.1, 0.001, 0.1, 0.1, 0.001, 0.1, 0.1, 0.001, 0.1, 0.01, 0.01, 0.1, 0.1]
# JOINT_DAMPING = [0.001]
# FORWARD_DIR_OFFSET = np.array([0, 0, 0])


# import numpy as np

# URDF_FILENAME = "a1/a1.urdf"

# REF_POS_SCALE = 0.825
# INIT_POS = np.array([0, 0, 0.32])
# INIT_ROT = np.array([0, 0, 0, 1.0])

# SIM_TOE_JOINT_IDS = [
#     5,  # right hand
#     15,  # right foot
#     10,  # left hand
#     20,  # left foot
# ]
# SIM_HIP_JOINT_IDS = [1, 11, 6, 16]
# SIM_ROOT_OFFSET = np.array([0, 0, -0.06])
# SIM_TOE_OFFSET_LOCAL = [
#     np.array([0, -0.05, 0.0]),
#     np.array([0, -0.05, 0.01]),
#     np.array([0, 0.05, 0.0]),
#     np.array([0, 0.05, 0.01])
# ]

# DEFAULT_JOINT_POSE = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
# JOINT_DAMPING = [0.1, 0.05, 0.01,
#                  0.1, 0.05, 0.01,
#                  0.1, 0.05, 0.01,
#                  0.1, 0.05, 0.01]

# FORWARD_DIR_OFFSET = np.array([0, 0, 0])
