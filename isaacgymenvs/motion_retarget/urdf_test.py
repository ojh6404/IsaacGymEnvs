import time
import numpy as np
import numpy.core.umath_tests as ut
from utils import BVH, Animation
from utils import pose3d

import pybullet
import pybullet_data as pd
from pybullet_utils import transformations

import config.bvh_cfg.bvh_cmu_config as bvh_cfg
import config.robot_cfg.khr_retarget_config_cmu as khr_cfg


# JOINT_NAMES = ['HIP_VIRTUAL_JOINT', 'SPINE_VIRTUAL_JOINT', 'TORSO_VIRTUAL_JOINT', 'HEAD_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LHAND_VIRTUAL_JOINT', 'RARM_JOINT0', 'RARM_JOINT1',
#                'RARM_JOINT2', 'RHAND_VIRTUAL_JOINT', 'LLEG_JOINT0', 'LLEG_JOINT1', 'LLEG_JOINT2', 'LLEG_JOINT3', 'LLEG_JOINT4', 'RLEG_JOINT0', 'RLEG_JOINT1', 'RLEG_JOINT2', 'RLEG_JOINT3', 'RLEG_JOINT4']
# JOINT_NAMES = ['HIP_VIRTUAL_JOINT', 'SPINE_VIRTUAL_JOINT', 'TORSO_VIRTUAL_JOINT', 'HEAD_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LHAND_VIRTUAL_JOINT', 'LARM_VIRTUAL_JOINT2', 'LARM_VIRTUAL_JOINT1', 'RARM_JOINT0', 'RARM_JOINT1',
#                'RARM_JOINT2', 'RHAND_VIRTUAL_JOINT', 'RARM_VIRTUAL_JOINT2', 'RARM_VIRTUAL_JOINT1', 'LLEG_JOINT0', 'LLEG_JOINT1', 'LLEG_JOINT2', 'LLEG_JOINT3', 'LLEG_JOINT4', 'RLEG_JOINT0', 'RLEG_JOINT1', 'RLEG_JOINT2', 'RLEG_JOINT3', 'RLEG_JOINT4']
JOINT_NAMES = ['HIP_VIRTUAL_JOINT', 'SPINE_VIRTUAL_JOINT', 'TORSO_VIRTUAL_JOINT', 'HEAD_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LHAND_VIRTUAL_JOINT', 'LARM_VIRTUAL_JOINT2', 'LARM_VIRTUAL_JOINT1', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RHAND_VIRTUAL_JOINT', 'RARM_VIRTUAL_JOINT2',
               'RARM_VIRTUAL_JOINT1', 'LLEG_JOINT0', 'LLEG_JOINT1', 'LLEG_JOINT2', 'LLEG_JOINT3', 'LLEG_JOINT4', 'LLEG_VIRTUAL_JOINT3', 'LLEG_VIRTUAL_JOINT2', 'LLEG_VIRTUAL_JOINT1', 'RLEG_JOINT0', 'RLEG_JOINT1', 'RLEG_JOINT2', 'RLEG_JOINT3', 'RLEG_JOINT4', 'RLEG_VIRTUAL_JOINT3', 'RLEG_VIRTUAL_JOINT2', 'RLEG_VIRTUAL_JOINT1']


def build_markers(num_markers):
    print(num_markers)
    marker_radius = 0.02
    markers_handle = []
    for i in range(num_markers):
        if (JOINT_NAMES[i] == 'HIP_VIRTUAL_JOINT') or (JOINT_NAMES[i] == 'TORSO_VIRTUAL_JOINT'):
            col = [0, 1, 0, 1]
        elif 'R' in JOINT_NAMES[i]:
            col = [0.9, 0, 0.7, 1]
        elif 'L' in JOINT_NAMES[i]:
            col = [0, 0, 0, 1]
        else:
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
        khr_cfg.GROUND_URDF_FILENAME, basePosition=[0., 0., 0.])

    # create actor
    robot = pybullet.loadURDF(khr_cfg.ROBOT_URDF_FILENAME, basePosition=np.array(
        [0, 0, 0.283]), baseOrientation=np.array([0, 0, 0, 1]))
    num_joints = pybullet.getNumJoints(robot)
    robot_joint_indices = {}
    for i in range(num_joints):
        joint_name = str(pybullet.getJointInfo(robot, i)[1], 'utf-8')
        # print(pybullet.getJointInfo(robot, i))
        robot_joint_indices[joint_name] = i

    print(robot_joint_indices)
    non_fixed_joint_indices = get_non_fixed_joint_indices(robot)

    print(non_fixed_joint_indices)
    non_fixed_joint_dict = {}
    for i in range(num_joints):
        joint_name = str(pybullet.getJointInfo(robot, i)[1], 'utf-8')
        if "HEAD" in joint_name or "VIRTUAL" in joint_name:
            continue
        # print(pybullet.getJointInfo(robot, i))
        non_fixed_joint_dict[joint_name] = i

    print('non fixed joint')
    print(non_fixed_joint_dict)

    joint_lower, joint_upper, joint_limit_range = get_joint_limits(robot)

    # print(robot_joint_indices)

    joint = "LARM_JOINT0"
    for k in range(num_joints):
        if k == non_fixed_joint_dict[joint]:
            pybullet.resetJointState(
                robot, non_fixed_joint_dict[joint], joint_lower[non_fixed_joint_indices.index(k)], 0.)
        else:
            continue

    # DEFAULT_JOINT_POS = [0] * len(non_fixed_joint_dict)
    # DEFAULT_JOINT_POS[0] = 1.
    # DEFAULT_JOINT_POS = [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, -0.5,
    #                      0, 0.3, 0.0, -0.5, 0, 0, -0.4, 0.8, -0.4, 0, 0, -0.4, 0.8, -0.4, 0]
    # print(len(DEFAULT_JOINT_POS))
    # print(DEFAULT_JOINT_POS)
    print(joint_upper)
    # joint = "LARM_JOINT0"
    # for k in range(num_joints):
    #     # if k == non_fixed_joint_dict[joint]:
    #     pybullet.resetJointState(
    #         robot, k, DEFAULT_JOINT_POS[k], 0.)
    # else:
    # continue
    print(robot_joint_indices.keys())
    # robot_joint_indices
    # {'HEAD_JOINT0': 0, 'LARM_JOINT0': 1, 'LARM_JOINT1': 2, 'LARM_JOINT2': 3,
    # 'LLEG_JOINT0': 4, 'LLEG_JOINT1': 5, 'LLEG_JOINT2': 6, 'LLEG_JOINT3': 7,
    # 'LLEG_JOINT4': 8, 'RARM_JOINT0': 9, 'RARM_JOINT1': 10, 'RARM_JOINT2': 11,
    # 'RLEG_JOINT0': 12, 'RLEG_JOINT1': 13, 'RLEG_JOINT2': 14, 'RLEG_JOINT3': 15,
    # 'RLEG_JOINT4': 16}

    # create marker to display reference motion
    num_markers = num_joints
    marker_ids = build_markers(num_markers)

    states = np.array(pybullet.getLinkStates(robot, list(
        range(num_joints)), computeForwardKinematics=True))[:, 4]

    ref_joint_pos = states
    while True:
        time_start = time.time()

        # print("Frame {:d}".format(f_idx))

        # ref_joint_pos = ref_joint_position
        # ref_joint_pos = np.reshape(ref_joint_pos, [-1, 3])
        # ref_joint_pos = pre_process_ref_joint_pos(ref_joint_pos)
        # ref_joint_pos += 1.0

        # if f_idx == 30:
        #   a = np.array(pybullet.getLinkState(robot, 15)[0])
        #   print(a.shape)
        #   a = a - np.array(pose[0:3])
        #   print(a)
        set_robot_joint_marker(robot, marker_ids)
        # set_maker_pos(ref_joint_pos, marker_ids)
        # update_camera(robot)

        time_end = time.time()
        sleep_dur = bvh_cfg.FRAME_DURATION - (time_end - time_start)
        sleep_dur = max(0, sleep_dur)

        time.sleep(sleep_dur * 2)
        # time.sleep(0.02)  # jp hack

    pybullet.disconnect()

    return


if __name__ == "__main__":
    main()
