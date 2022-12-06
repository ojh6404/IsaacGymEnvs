import time
import numpy as np
import numpy.core.umath_tests as ut
from utils import BVH, Animation
from utils import pose3d

import pybullet
import pybullet_data as pd
from pybullet_utils import transformations

import config.bvh_cfg.bvh_cmu_config as bvh_cfg
import config.robot_cfg.atlas_retarget_config_cmu as atlas_cfg


def test_build_markers(num_markers):
    marker_radius = 0.03
    markers_handle = []
    for i in range(num_markers):
        if (bvh_cfg.BVH_JOINT_NAMES[i] == 'Hips') or (bvh_cfg.BVH_JOINT_NAMES[i] == 'Spine')\
                or (bvh_cfg.BVH_JOINT_NAMES[i] == 'Spine1') or (bvh_cfg.BVH_JOINT_NAMES[i] == 'Neck'):
            col = [0, 1, 0, 1]
        elif 'R' in bvh_cfg.BVH_JOINT_NAMES[i]:
            col = [0.9, 0, 0.7, 1]
        elif 'L' in bvh_cfg.BVH_JOINT_NAMES[i]:
            col = [0, 0, 0, 1]
        else:
            col = [1, 1, 0, 1]
        # col = [0, 1, 0, 1]
        virtual_shape_id = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                      radius=marker_radius,
                                                      rgbaColor=col)
        body_id = pybullet.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=-1,
                                           baseVisualShapeIndex=virtual_shape_id,
                                           basePosition=[0, 0, 0],
                                           useMaximalCoordinates=True)
        markers_handle.append(body_id)
    return markers_handle


def build_markers(num_markers):
    marker_radius = 0.07
    markers_handle = []
    for i in range(num_markers):
        if (i == 9) or (i == 86):
            col = [0, 1, 0, 1]
        # elif 'R' in bvh_cfg.BVH_JOINT_NAMES[i]:
        #     col = [0.9, 0, 0.7, 1]
        # elif 'L' in bvh_cfg.BVH_JOINT_NAMES[i]:
        #     col = [0, 0, 0, 1]
        else:
            col = [1, 1, 0, 1]
        col = [1, 1, 0, 1]
        virtual_shape_id = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                      radius=marker_radius,
                                                      rgbaColor=col)
        body_id = pybullet.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=-1,
                                           baseVisualShapeIndex=virtual_shape_id,
                                           basePosition=[0, 0, 0],
                                           useMaximalCoordinates=True)
        markers_handle.append(body_id)
    return markers_handle


def get_joint_limits(robot):

    num_joints = pybullet.getNumJoints(robot)
    joint_lower_bound = []
    joint_upper_bound = []
    joint_limit_range = []

    for i in range(num_joints):
        joint_info = pybullet.getJointInfo(robot, i)
        joint_type = joint_info[2]
        if joint_type == pybullet.JOINT_PRISMATIC or joint_type == pybullet.JOINT_REVOLUTE:
            joint_lower_bound.append(joint_info[8])
            joint_upper_bound.append(joint_info[9])
            joint_limit_range.append(joint_info[9] - joint_info[8])
    return joint_lower_bound, joint_upper_bound, joint_limit_range


def set_maker_pos(marker_pos, marker_ids):
    num_markers = len(marker_ids)
    # print(marker_pos.shape[0])
    # print(num_markers)
    assert(num_markers == marker_pos.shape[0])

    for i in range(num_markers):
        curr_id = marker_ids[i]
        curr_pos = marker_pos[i]

        pybullet.resetBasePositionAndOrientation(
            curr_id, curr_pos, np.array([0, 0, 0, 1]))

    return


def set_robot_joint_marker(robot, marker_ids):

    num_joints = pybullet.getNumJoints(robot)

    robot_joint_pos = np.array(pybullet.getLinkStates(robot, list(
        range(num_joints)), computeForwardKinematics=True))[:, 4]
    # print('debug')
    # print(robot_joint_pos[0])
    set_maker_pos(robot_joint_pos, marker_ids)


def get_non_fixed_joint_indices(robot):
    num_joints = pybullet.getNumJoints(robot)
    non_fixed_joint_indices = []
    for i in range(num_joints):
        joint_type = pybullet.getJointInfo(robot, i)[2]
        if joint_type is not 4:
            non_fixed_joint_indices.append(i)
    return non_fixed_joint_indices


def main():
    # build world
    pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pd.getDataPath())
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, -0)
    ground = pybullet.loadURDF(
        atlas_cfg.GROUND_URDF_FILENAME, basePosition=[0., 0., 0.])

    frames = np.load("./test.npz")
    retarget_pos = frames["retarget_frames"]
    ref_joint_pos_test = frames["ref_joint_pos"]
    ref_joint_pos_test = np.array(ref_joint_pos_test)
    print(ref_joint_pos_test.shape)
    init_height = ref_joint_pos_test[0, 0, 2]

    # create actor
    robot = pybullet.loadURDF(atlas_cfg.ROBOT_URDF_FILENAME, basePosition=np.array(
        [0, 0, init_height]), baseOrientation=np.array([0, 0, 0, 1]))
    num_joints = pybullet.getNumJoints(robot)
    robot_joint_indices = {}
    for i in range(num_joints):
        joint_name = str(pybullet.getJointInfo(robot, i)[1], 'utf-8')
        # print(pybullet.getJointInfo(robot, i))
        robot_joint_indices[joint_name] = i

    #{'back_bkz': 0, 'back_bky': 1, 'back_bkx': 2, 'l_arm_shz': 3, 'l_arm_shx': 4, 'l_arm_ely': 5, 'l_arm_elx': 6, 'l_arm_wry': 7, 'l_arm_wrx': 8, 'l_arm_wry2': 9, 'neck_ry': 10, 'r_arm_shz': 11, 'r_arm_shx': 12, 'r_arm_ely': 13, 'r_arm_elx': 14, 'r_arm_wry': 15, 'r_arm_wrx': 16, 'r_arm_wry2': 17, 'l_leg_hpz': 18, 'l_leg_hpx': 19, 'l_leg_hpy': 20, 'l_leg_kny': 21, 'l_leg_aky': 22, 'l_leg_akx': 23, 'r_leg_hpz': 24, 'r_leg_hpx': 25, 'r_leg_hpy': 26, 'r_leg_kny': 27, 'r_leg_aky': 28, 'r_leg_akx': 29}
    non_fixed_joint_indices = get_non_fixed_joint_indices(robot)
    print(robot_joint_indices)

    joint_lower, joint_upper, joint_limit_range = get_joint_limits(robot)

    # for k in range(len(non_fixed_joint_indices)):
    #     pybullet.resetJointState(
    #         robot, non_fixed_joint_indices[k], joint_lower[k], 0.)
    #
    # {'torso_waist_y': 0, 'torso_waist_p': 1, 'torso_waist_p2': 2, 'head_neck_y': 3, 'head_neck_p': 4, 'rx78_Null_013_joint': 5, 'rx78_Null_012_joint': 6, 'rx78_Null_011_joint': 7, 'rx78_Null_010_joint': 8, 'rx78_Null_009_joint': 9, 'rx78_Null_007_joint': 10, 'rx78_Null_008_joint': 11, 'torso_rthrust_p': 12, 'torso_rthrust_r': 13, 'torso_lthrust_p': 14, 'torso_lthrust_r': 15, 'larm_shoulder_p': 16, 'larm_shoulder_r': 17, 'larm_shoulder_y': 18, 'larm_elbow_p': 19, 'larm_elbow_p2': 20, 'larm_wrist_y': 21, 'larm_wrist_r': 22, 'larm_gripper_middle0_mimic': 23, 'larm_gripper_middle1_mimic': 24, 'larm_gripper_middle2_mimic': 25, 'larm_gripper_ring0_mimic': 26, 'larm_gripper_ring1_mimic': 27, 'larm_gripper_ring2_mimic': 28, 'larm_gripper_little0_mimic': 29, 'larm_gripper_little1_mimic': 30, 'larm_gripper_little2_mimic': 31, 'larm_gripper_index0_mimic': 32, 'larm_gripper_index1_mimic': 33, 'larm_gripper_index2_mimic': 34, 'larm_gripper': 35, 'larm_gripper_thumb1_mimic': 36, 'larm_gripper_thumb2_mimic': 37, 'rx78_Null_065_joint': 38, 'rx78_Null_048_joint': 39, 'rarm_shoulder_p': 40, 'rarm_shoulder_r': 41, 'rarm_shoulder_y': 42, 'rarm_elbow_p': 43, 'rarm_elbow_p2': 44, 'rarm_wrist_y': 45, 'rarm_wrist_r': 46, 'rarm_gripper': 47, 'rarm_gripper_thumb1_mimic': 48, 'rarm_gripper_thumb2_mimic': 49, 'rarm_gripper_middle0_mimic': 50, 'rarm_gripper_middle1_mimic': 51, 'rarm_gripper_middle2_mimic': 52, 'rarm_gripper_index0_mimic': 53, 'rarm_gripper_index1_mimic': 54, 'rarm_gripper_index2_mimic': 55, 'rarm_gripper_little0_mimic': 56, 'rarm_gripper_little1_mimic': 57, 'rarm_gripper_little2_mimic': 58, 'rarm_gripper_ring0_mimic': 59, 'rarm_gripper_ring1_mimic': 60, 'rarm_gripper_ring2_mimic': 61, 'rx78_Null_081_joint': 62, 'rx78_Null_082_joint': 63, 'rx78_Null_083_joint': 64, 'rleg_crotch_p_front_mimic': 65, 'rleg_crotch_p_back_mimic': 66, 'rleg_crotch_r_mimic': 67, 'lleg_crotch_p_back_mimic': 68, 'lleg_crotch_p_front_mimic': 69, 'lleg_crotch_r_mimic': 70, 'lleg_crotch_p': 71, 'lleg_crotch_r': 72, 'lleg_crotch_y': 73, 'lleg_knee_p': 74, 'lleg_knee_p2': 75, 'lleg_ankle_p': 76, 'lleg_ankle_r_mimic': 77, 'lleg_ankle_r': 78, 'rx78_Null_042_joint': 79, 'rx78_Null_043_joint': 80, 'rx78_Null_044_joint': 81, 'rx78_Null_045_joint': 82, 'lleg_ankle_p_mimic': 83, 'rleg_crotch_p': 84, 'rleg_crotch_r': 85, 'rleg_crotch_y': 86, 'rleg_knee_p': 87, 'rleg_knee_p2': 88, 'rleg_ankle_p': 89, 'rleg_ankle_r_mimic': 90, 'rleg_ankle_r': 91, 'rx78_Null_092_joint': 92, 'rx78_Null_093_joint': 93, 'rx78_Null_094_joint': 94, 'rx78_Null_095_joint': 95, 'rleg_ankle_p_mimic': 96}

    # create marker to display reference motion
    num_markers = num_joints
    marker_ids = build_markers(num_markers)

    test_num_markers = len(bvh_cfg.BVH_JOINT_NAMES)
    test_marker_ids = test_build_markers(test_num_markers)

    states = np.array(pybullet.getLinkStates(robot, list(
        range(num_joints)), computeForwardKinematics=True))[:, 4]

    print(list(robot_joint_indices.keys()))

    test_marker_idx = build_markers(1)

    ref_joint_pos = states
    f = 0
    while True:
        time_start = time.time()

        f_idx = f % ref_joint_pos_test.shape[0]
        set_robot_joint_marker(robot, marker_ids)

        # set_maker_pos(np.array([0, 5, 12.5]).reshape(1, 3), test_marker_idx)

        set_maker_pos(ref_joint_pos_test[f_idx] *
                      atlas_cfg.REF_POS_SCALE, test_marker_ids)

        time_end = time.time()
        sleep_dur = bvh_cfg.FRAME_DURATION - (time_end - time_start)
        sleep_dur = max(0, sleep_dur)
        f += 1
        # time.sleep(sleep_dur)
        input()

    pybullet.disconnect()

    return


if __name__ == "__main__":
    main()
