import time
import numpy as np
import numpy.core.umath_tests as ut
from utils import BVH, Animation
from utils import pose3d

import pybullet
import pybullet_data as pd
from pybullet_utils import transformations

import config.bvh_cfg.bvh_cmu_config as bvh_cfg
import config.robot_cfg.khr22_retarget_config_cmu as robot_cfg


def build_markers(num_markers):
    JOINT_NAMES = robot_cfg.JOINT_NAMES
    print(num_markers)
    marker_radius = 0.01
    markers_handle = []
    for i in range(num_markers):
        if "VIRTUAL" in JOINT_NAMES[i] or "HEAD" in JOINT_NAMES[i]:
            col = [0, 0, 1, 1]  # blue : VIRTUAL JOINT and Fixed HEAD JOINT
        elif 'TORSO' in JOINT_NAMES[i]:
            col = [1, 1, 0, 1]  # yellow : TORSO JOINT
        elif 'R' in JOINT_NAMES[i]:
            col = [0.9, 0, 0.7, 1]  # pink : Right JOINT
        elif 'L' in JOINT_NAMES[i]:
            col = [0, 1, 0, 1]  # green : Left JOINT
        else:
            col = [0, 0, 0, 1]  # black : other
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


def set_marker_pos(marker_pos, marker_ids):
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

    set_marker_pos(robot_joint_pos, marker_ids)


def get_non_fixed_joint_indices(robot):
    num_joints = pybullet.getNumJoints(robot)
    non_fixed_joint_indices = []
    for i in range(num_joints):
        joint_type = pybullet.getJointInfo(robot, i)[2]
        if joint_type is not 4:
            non_fixed_joint_indices.append(i)
    return non_fixed_joint_indices


def build_world():
    pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pd.getDataPath())
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, -9.8)

    # create ground
    ground = pybullet.loadURDF("plane_implicit.urdf",
                               basePosition=[0., 0., 0.])

    # create actor
    robot = pybullet.loadURDF(robot_cfg.ROBOT_URDF_FILENAME, basePosition=np.array(
        robot_cfg.INIT_POS), baseOrientation=robot_cfg.INIT_QUAT)

    return robot, ground


def get_robot_joint_indices(robot):
    robot_joint_indices = {}
    for i in range(pybullet.getNumJoints(robot)):
        joint_name = str(pybullet.getJointInfo(robot, i)[1], 'utf-8')
        # print(pybullet.getJointInfo(robot, i))
        robot_joint_indices[joint_name] = i

    return robot_joint_indices


def main():
    # build world
    robot, ground = build_world()
    num_joints = pybullet.getNumJoints(robot)
    robot_joint_indices = get_robot_joint_indices(robot)
    non_fixed_joint_indices = get_non_fixed_joint_indices(robot)

    print("robot's joint :", robot_joint_indices)
    print("robot's non-fixed-joint :", non_fixed_joint_indices)

    non_fixed_joint_dict = {}
    for i in range(num_joints):
        joint_name = str(pybullet.getJointInfo(robot, i)[1], 'utf-8')
        if "HEAD" in joint_name or "VIRTUAL" in joint_name:
            continue
        # print(pybullet.getJointInfo(robot, i))
        non_fixed_joint_dict[joint_name] = i

    # print('non fixed joint')
    # print(non_fixed_joint_dict)

    joint_lower, joint_upper, joint_limit_range = get_joint_limits(robot)

    # TODO : check joint limit
    joint = "LARM_JOINT2"
    for k in range(num_joints):
        if k == non_fixed_joint_dict[joint]:
            pybullet.resetJointState(
                robot, non_fixed_joint_dict[joint], joint_upper[non_fixed_joint_indices.index(k)], 0.)
        else:
            continue

    print("robot joint indices")
    print(robot_joint_indices.keys())

    # create marker to display reference motion
    num_markers = num_joints
    marker_ids = build_markers(num_markers)

    states = np.array(pybullet.getLinkStates(robot, list(
        range(num_joints)), computeForwardKinematics=True))[:, 4]

    ref_joint_pos = states
    while True:
        time_start = time.time()

        set_robot_joint_marker(robot, marker_ids)
        # set_marker_pos(ref_joint_pos, marker_ids)

        time_end = time.time()
        sleep_dur = bvh_cfg.FRAME_DURATION - (time_end - time_start)
        sleep_dur = max(0, sleep_dur)

        time.sleep(sleep_dur * 2)
        # time.sleep(0.02)  # jp hack

    pybullet.disconnect()

    return


if __name__ == "__main__":
    main()
