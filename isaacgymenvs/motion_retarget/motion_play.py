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


def build_markers(num_markers):
    marker_radius = 0.005
    markers_handle = []
    for i in range(num_markers):
        if (bvh_cfg.BVH_JOINT_NAMES[i] == 'Hips') or (bvh_cfg.BVH_JOINT_NAMES[i] == 'Spine')\
                or (bvh_cfg.BVH_JOINT_NAMES[i] == 'Spine1') or (bvh_cfg.BVH_JOINT_NAMES[i] == 'Neck'):
            col = [0, 1, 0, 1]
        elif 'L' in bvh_cfg.BVH_JOINT_NAMES[i]:
            col = [0.9, 0, 0.7, 1]
        elif 'R' in bvh_cfg.BVH_JOINT_NAMES[i]:
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


def set_marker_pos(marker_pos, marker_ids):
    num_markers = len(marker_ids)
    assert(num_markers == marker_pos.shape[0])

    for i in range(num_markers):
        curr_id = marker_ids[i]
        curr_pos = marker_pos[i]

        pybullet.resetBasePositionAndOrientation(
            curr_id, curr_pos, np.array([0, 0, 0, 1]))

    return


def get_root_position(pose):
    return pose[0:3]


def get_root_quaternion(pose):
    return pose[3:7]


def get_joint_position(pose):
    return pose[7:]


def set_root_position(root_position, pose):
    pose[0:3] = root_position
    return pose


def set_root_quaternion(root_quaternion, pose):
    pose[3:7] = root_quaternion
    return pose


def set_joint_position(joint_position, pose):
    pose[7:] = joint_position
    return pose


def set_pose(robot, pose):
    num_joints = pybullet.getNumJoints(robot)
    root_position = get_root_position(pose)
    root_quaternion = get_root_quaternion(pose)

    pybullet.resetBasePositionAndOrientation(
        robot, root_position, root_quaternion)

    for j in range(num_joints):
        j_info = pybullet.getJointInfo(robot, j)
        j_pose_idx = j_info[3]

        j_state = pybullet.getJointStateMultiDof(robot, j)
        j_pose_size = len(j_state[0])
        j_vel_size = len(j_state[1])

        if j_pose_size > 0:
            j_pose = pose[j_pose_idx:(j_pose_idx + j_pose_size)]
            j_vel = np.zeros(j_vel_size)
            pybullet.resetJointStateMultiDof(robot, j, j_pose, j_vel)
    return


def build_world():
    pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pd.getDataPath())
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, -9.8)

    # create ground
    ground = pybullet.loadURDF(
        "plane_implicit.urdf", basePosition=[0., 0., 0.])

    # create actor
    robot = pybullet.loadURDF(khr_cfg.ROBOT_URDF_FILENAME, basePosition=np.array(
        [0, 0, 0.3]), baseOrientation=np.array([0, 0, 0, 1]))

    return robot, ground


def main():
    # build world
    robot, ground = build_world()

    # get robot joints info
    num_joints = pybullet.getNumJoints(robot)

    robot_joint_indices = {}
    for i in range(num_joints):
        joint_name = str(pybullet.getJointInfo(robot, i)[1], 'utf-8')
        # print(pybullet.getJointInfo(robot, i))
        robot_joint_indices[joint_name] = i

    # load frame of retargeted motion and reference motion data
    frames = np.load("./result/output/07_08.npz")
    retarget_pos = frames["retarget_frames"]
    ref_joint_pos = frames["ref_joint_pos"]

    # create marker to display reference motion
    num_frames = ref_joint_pos.shape[0]
    num_markers = ref_joint_pos.shape[1]
    marker_ids = build_markers(num_markers)

    f = 0

    while True:
        time_start = time.time()

        f_idx = f % num_frames

        pose = retarget_pos[f_idx]
        set_pose(robot, pose)

        set_marker_pos(ref_joint_pos[f_idx], marker_ids)

        f += 1

        time_end = time.time()
        sleep_dur = bvh_cfg.FRAME_DURATION - (time_end - time_start)
        sleep_dur = max(0, sleep_dur)

        time.sleep(sleep_dur * 1.0)
        # time.sleep(0.02)  # jp hack

    pybullet.disconnect()

    return


if __name__ == '__main__':
    main()
