import numpy as np

# -----------------------------------------SFU Skeleton Configuration-----------------------
# FILE_DIR = 'bvh_data/SFU_Motion_Capture_Database/walking/'
FILE_DIR = 'bvh_data/CMU/016/'
# FILE_DIR = 'bvh_data/CMU/139/'
# file names list which should be retargeted
# FILE_LIST = ["0007_Walking001.bvh"]
FILE_LIST = ["16_35.bvh"]
# FILE_LIST = ["139_16.bvh"]
SAVE_TEMP_FILE = True
TEMP_FILE_DIR = "result/temp/"
OUT_FILE_DIR = "result/output/"
# scale the global position of all joints during calculating global reference position for joints cm--->m
POSITION_SCALING = 0.01
Y_UP_AXIS = True  # the coordinate used in bvh file
LEFT_HAND_COORDINATE = False
CAL_END_EFFECTOR = False
FRAME_DURATION = 0.008333

BVH_JOINT_NAMES = ['Hips', 'LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                   'LeftToeBase', 'RHipJoint', 'RightUpLeg', 'RightLeg', 'RightFoot',
                   'RightToeBase', 'LowerBack', 'Spine', 'Spine1', 'Neck',
                   'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm',
                   'LeftHand', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb', 'RightShoulder',
                   'RightArm', 'RightForeArm', 'RightHand', 'RightFingerBase', 'RightHandIndex1', 'RThumb']


#

# BVH_JOINT_NAMES = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
#                    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine',
#                    'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm',
#                    'LeftForeArm', 'LeftHand', 'LeftHandThumb', 'L_Wrist_End',
#                    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb',
#                    'R_Wrist_End']
BVH_ROOT_HEIGHT = 0.157 + 0.154 + 0.015
BVH_CHESET_LENGTH = 0.103 + 0.078
BVH_HAND_LENGTH = 0.1094 + 0.0852

# LH_JOINT_IDX =
# RH_JOINT_IDX =
# LF_JOINT_IDX =
# RF_JOINT_IDX =

# LH_END_OFFSET = np.array([, 1], dtype=np.float64).reshape(4, 1)
# RH_END_OFFSET = np.array([, 1], dtype=np.float64).reshape(4, 1)
# LF_END_OFFSET = np.array([, 1], dtype=np.float64).reshape(4, 1)
# RF_END_OFFSET = np.array([, 1], dtype=np.float64).reshape(4, 1)
