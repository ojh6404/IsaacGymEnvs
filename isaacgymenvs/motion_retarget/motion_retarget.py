import time
import numpy as np
from utils import BVH, Animation
from utils import pose3d
from utils.Quaternions import Quaternions

import pybullet
import pybullet_data as pd
from pybullet_utils import transformations


import config.bvh_cfg.bvh_cmu_config as bvh_cfg
import config.robot_cfg.khr_retarget_config_cmu as khr_cfg


# DEFAULT_JOINT_POS = [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, -0.5,
#                      0, 0.3, 0.0, -0.5, 0, 0, -0.4, 0.8, -0.4, 0, 0, -0.4, 0.8, -0.4, 0]


def set_joint_pos_origin(joint_global_pos):
    """
    eliminate the offset of motions
    set the motions start at the origin
    """
    pos_data_origin = joint_global_pos.copy()
    if bvh_cfg.Y_UP_AXIS:
        x_offset = joint_global_pos[0, 0, 0]
        z_offset = joint_global_pos[0, 0, 2]
        for i in range(pos_data_origin.shape[0]):
            for j in range(pos_data_origin.shape[1]):
                for k in range(pos_data_origin.shape[2]):
                    a = k % 3
                    if a == 0:
                        pos_data_origin[i, j, k] -= x_offset
                    elif a == 2:
                        pos_data_origin[i, j, k] -= z_offset
                    else:
                        pos_data_origin[i, j, k] = pos_data_origin[i, j, k]
    else:
        raise NotImplementedError
    return pos_data_origin


def get_joint_global_pos(bvh_file):
    """
    get the joint position in global frame
    scale the raw position data by POSITION_SCALING

    -----------------
    Returns
    positions : (F, J, 3) ndarray
        Positions for every frame F
        and joint position J

    if cal_end_effector:
        positions : (F, J+N, 3) ndarray, N: number of end effector
    """
    anim, joint_names, frame_interval = BVH.load(bvh_file)
    joint_position = Animation.positions_global(anim)

    joint_position *= bvh_cfg.POSITION_SCALING
    joint_position = set_joint_pos_origin(joint_position)
    return joint_position, joint_names


def build_markers(num_markers, col=[0., 0., 0., 1.], size=0.01):
    marker_radius = size
    markers_handle = []
    for i in range(num_markers):
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


def get_joint_limits2(robot):

    num_joints = pybullet.getNumJoints(robot)
    joint_lower_bound = []
    joint_upper_bound = []
    joint_limit_range = []

    for i in range(num_joints):
        joint_info = pybullet.getJointInfo(robot, i)
        joint_type = joint_info[2]
        if joint_type == pybullet.JOINT_PRISMATIC or joint_type == pybullet.JOINT_REVOLUTE or joint_type == pybullet.JOINT_FIXED:
            joint_lower_bound.append(joint_info[8])
            joint_upper_bound.append(joint_info[9])
            joint_limit_range.append(joint_info[9] - joint_info[8])
    return joint_lower_bound, joint_upper_bound, joint_limit_range


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


def set_marker_pos(marker_pos, marker_ids):
    num_markers = len(marker_ids)
    assert(num_markers == marker_pos.shape[0])

    for i in range(num_markers):
        curr_id = marker_ids[i]
        curr_pos = marker_pos[i]

        pybullet.resetBasePositionAndOrientation(
            curr_id, curr_pos, np.array([0, 0, 0, 1]))

    return


def pre_process_ref_joint_pos(ref_joint_pos):
    """
    1. align the coordinate: most bvh file use right-hand and y-up axis, while khr urdf and pybullet use z-up with right-hand coordinate
    2. roughly scale the position to align the skeleton size with robot size
    3. ref_joint_pos: just one frame data, (J, 3)
    BVH : Left-Handed coord and Y_UP_AXIS  -> Pybullet : Right_handed coord and Z_UP_AXIS
    """
    proc_pos = ref_joint_pos.copy()
    num_pos = ref_joint_pos.shape[0]  # number positions(joints)
    # print("anim_position", proc_pos[0])
    if bvh_cfg.Y_UP_AXIS and (not bvh_cfg.LEFT_HAND_COORDINATE):
        for i in range(num_pos):
            curr_pos = proc_pos[i]
            curr_pos = pose3d.QuaternionRotatePoint(
                curr_pos, transformations.quaternion_from_euler(0.5 * np.pi, 0, 0.5 * np.pi, axes='sxyz'))
            # curr_pos = pose3d.QuaternionRotatePoint(curr_pos, transformations.quaternion_from_euler(0, 0, -0.5 * np.pi))
            curr_pos = curr_pos * khr_cfg.REF_POS_SCALE
            proc_pos[i] = curr_pos

    else:
        raise NotImplementedError
    # print("prepocess", proc_pos[0])
    return proc_pos


def retarget_root_pose(ref_joint_pos):
    """
    retarget the root pose:
    1. use the left_up_leg and right_up_leg as directions y
    2. use spine1 and hip as directions z
    3. cross(y,z) ----> get x direction
    4. cross(z, x) ----> get new y direction
    """
    hip_index = bvh_cfg.BVH_JOINT_NAMES.index('Hips')
    spine1_index = bvh_cfg.BVH_JOINT_NAMES.index('Spine1')
    left_upleg_index = bvh_cfg.BVH_JOINT_NAMES.index('LeftUpLeg')
    right_upleg_index = bvh_cfg.BVH_JOINT_NAMES.index('RightUpLeg')

    y_direction = ref_joint_pos[left_upleg_index] - \
        ref_joint_pos[right_upleg_index]
    y_direction /= np.linalg.norm(y_direction)
    z_direction = ref_joint_pos[spine1_index] - ref_joint_pos[hip_index]
    z_direction /= np.linalg.norm(z_direction)
    # TODO: add some offset manually if the retargeted result are not very nature

    x_direction = np.cross(y_direction, z_direction)
    x_direction /= np.linalg.norm(x_direction)

    # make sure the y direction is vertical with the other two directions
    y_direction = np.cross(z_direction, x_direction)
    y_direction /= np.linalg.norm(y_direction)

    rotation_matrix = np.array([[x_direction[0], y_direction[0], z_direction[0], 0],
                                [x_direction[1], y_direction[1], z_direction[1], 0],
                                [x_direction[2], y_direction[2], z_direction[2], 0],
                                [0,              0,              0,              1]])
    root_quaternion = transformations.quaternion_from_matrix(rotation_matrix)

    # root_position = ref_joint_pos[0]
    root_position = ref_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Hips')]
    # print("root_position",root_position)
    root_height_scale = khr_cfg.KHR_ROOT_HEIGHT / bvh_cfg.BVH_ROOT_HEIGHT
    # root_position = root_position * root_height_scale  # TODO: scale z or scale x, y and z?
    # root_position[-1] = root_position[-1] + khr_cfg.KHR_ROOT_HEIGHT
    return root_position, root_quaternion


def get_non_fixed_joint_indices(robot):
    num_joints = pybullet.getNumJoints(robot)
    non_fixed_joint_indices = []
    for i in range(num_joints):
        joint_type = pybullet.getJointInfo(robot, i)[2]
        if joint_type != 4:
            non_fixed_joint_indices.append(i)
    return non_fixed_joint_indices


def scale_ref_pos(robot, ref_joint_pos):
    """
    Hips : HIP_VIRTUAL_JOINT (hip)
    Spine : SPINE_VIRTUAL_JOINT (spine)
    Spine1 : TORSO_VIRTUAL_JOINT (torso)
    Neck : HEAD_JOINT0 (neck, end)

    Spine1 : TORSO_VIRTUAL_JOINT (torso)
    Shoulder : ARM_JOINT0 (shoulder)
    Arm : ARM_JOINT1 (shoulder)
    ForeArm : ARM_JOINT2 (elbow)
    Hand : HAND_VIRTUAL_JOINT (hand, end-effector)

    Hip (HipJoint) : HIP_VIRTUAL_JOINT (hip)
    UpLeg : LEG_JOINT0 (crotch)
    Leg : LEG_JOINT2 (knee)
    Foot : LEG_JOINT4 (foot , end)
    """

    robot_joint_indices = get_robot_joint_indices(robot)
    scaled_joint_pos = ref_joint_pos.copy()

    non_fixed_joint_indices = get_non_fixed_joint_indices(robot)
    joint_lower_bound, joint_upper_bound, joint_limit_range = get_joint_limits(
        robot)

    robot_num_joints = len(robot_joint_indices)

    # align root pose by cross product each frame
    root_pos, root_quat = retarget_root_pose(ref_joint_pos)
    # pybullet.resetBasePositionAndOrientation(robot, root_pos, root_quater)

    # reference joint position from frame
    ref_hip_pos = ref_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Hips')]
    ref_spine_pos = ref_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Spine')]
    ref_torso_pos = ref_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Spine1')]
    ref_neck_pos = ref_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Neck')]
    ref_neck1_pos = ref_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Neck1')]
    ref_head_pos = ref_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Head')]

    ref_shoulder1_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftShoulder'), bvh_cfg.BVH_JOINT_NAMES.index('RightShoulder')]]
    ref_shoulder2_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftArm'), bvh_cfg.BVH_JOINT_NAMES.index('RightArm')]]
    ref_elbow_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftForeArm'), bvh_cfg.BVH_JOINT_NAMES.index('RightForeArm')]]
    ref_hand_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftHand'), bvh_cfg.BVH_JOINT_NAMES.index('RightHand')]]
    ref_finger1_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftFingerBase'), bvh_cfg.BVH_JOINT_NAMES.index('RightFingerBase')]]
    ref_finger2_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftHandIndex1'), bvh_cfg.BVH_JOINT_NAMES.index('RightHandIndex1')]]
    ref_thumb_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LThumb'), bvh_cfg.BVH_JOINT_NAMES.index('RThumb')]]

    ref_hips_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LHipJoint'), bvh_cfg.BVH_JOINT_NAMES.index('RHipJoint')]]
    ref_crotch_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftUpLeg'), bvh_cfg.BVH_JOINT_NAMES.index('RightUpLeg')]]
    ref_knee_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftLeg'), bvh_cfg.BVH_JOINT_NAMES.index('RightLeg')]]
    ref_foot_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftFoot'), bvh_cfg.BVH_JOINT_NAMES.index('RightFoot')]]
    ref_toe_pos = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftToeBase'), bvh_cfg.BVH_JOINT_NAMES.index('RightToeBase')]]

    # calculate vector between reference joint pos
    ref_hip_to_spine_pos_delta = ref_spine_pos - ref_hip_pos
    ref_spine_to_torso_pos_delta = ref_torso_pos - ref_spine_pos
    ref_torso_to_neck_pos_delta = ref_neck_pos - ref_torso_pos
    ref_neck_to_neck1_pos_delta = ref_neck1_pos - ref_neck_pos
    ref_neck1_to_head_pos_delta = ref_head_pos - ref_neck1_pos

    ref_torso_to_shoulder1_pos_delta = ref_shoulder1_pos - ref_torso_pos
    ref_shoulder1_to_shoulder2_pos_delta = ref_shoulder2_pos - ref_shoulder1_pos
    ref_shoulder2_to_elbow_pos_delta = ref_elbow_pos - ref_shoulder2_pos
    ref_elbow_to_hand_pos_delta = ref_hand_pos - ref_elbow_pos
    ref_hand_to_finger1_pos_delta = ref_finger1_pos - ref_hand_pos
    ref_finger1_to_finger2_pos_delta = ref_finger2_pos - ref_finger1_pos
    ref_finger2_to_thumb_pos_delta = ref_thumb_pos - ref_finger2_pos

    ref_hips_to_crotch_pos_delta = ref_crotch_pos - ref_hips_pos
    ref_crotch_to_knee_pos_delta = ref_knee_pos - ref_crotch_pos
    ref_knee_to_foot_pos_delta = ref_foot_pos - ref_knee_pos
    ref_foot_to_toe_pos_delta = ref_toe_pos - ref_foot_pos

    # scale upper body
    scaled_hip_to_spine_pos_delta = ref_hip_to_spine_pos_delta * \
        khr_cfg.HIP_TO_SPINE_SCALE
    scaled_spine_to_torso_pos_delta = ref_spine_to_torso_pos_delta * \
        khr_cfg.SPINE_TO_TORSO_SCALE
    scaled_torso_to_neck_pos_delta = ref_torso_to_neck_pos_delta * \
        khr_cfg.TORSO_TO_NECK_SCALE
    scaled_neck_to_neck1_pos_delta = ref_neck_to_neck1_pos_delta * \
        khr_cfg.NECK_TO_NECK1_SCALE
    scaled_neck1_to_head_pos_delta = ref_neck1_to_head_pos_delta * \
        khr_cfg.NECK1_TO_HEAD_SCALED

    scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index(
        'Spine')] = ref_hip_pos + scaled_hip_to_spine_pos_delta
    scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index(
        'LowerBack')] = scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Spine')]
    scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index(
        'Spine1')] = scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Spine')] + scaled_spine_to_torso_pos_delta
    scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index(
        'Neck')] = scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Spine1')] + scaled_torso_to_neck_pos_delta
    scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index(
        'Neck1')] = scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Neck')] + scaled_neck_to_neck1_pos_delta
    scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index(
        'Head')] = scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Neck1')] + scaled_neck1_to_head_pos_delta

    scaled_torso_to_shoulder1_pos_delta = ref_torso_to_shoulder1_pos_delta * \
        khr_cfg.TORSO_TO_SHOULDER1_SCALE
    scaled_shoulder1_to_shoulder2_pos_delta = ref_shoulder1_to_shoulder2_pos_delta * \
        khr_cfg.SHOULDER1_TO_SHOULDER2_SCALE
    scaled_shoulder2_to_elbow_pos_delta = ref_shoulder2_to_elbow_pos_delta * \
        khr_cfg.SHOULDER2_TO_ELBOW_SCALE
    scaled_elbow_to_hand_pos_delta = ref_elbow_to_hand_pos_delta * \
        khr_cfg.ELBOW_TO_HAND_SCALE
    scaled_hand_to_finger1_pos_delta = ref_hand_to_finger1_pos_delta * \
        khr_cfg.HAND_TO_FINGER1_SCALE
    scaled_finger1_to_finger2_pos_delta = ref_finger1_to_finger2_pos_delta * \
        khr_cfg.FINGER1_TO_FINGER2_SCALE
    scaled_finger2_to_thumb_pos_delta = ref_finger2_to_thumb_pos_delta * \
        khr_cfg.FIGNER2_TO_THUMB_SCALE

    scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftShoulder'), bvh_cfg.BVH_JOINT_NAMES.index('RightShoulder')]] = scaled_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index('Spine1')] + scaled_torso_to_shoulder1_pos_delta
    scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftArm'), bvh_cfg.BVH_JOINT_NAMES.index('RightArm')]] = scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
            'LeftShoulder'), bvh_cfg.BVH_JOINT_NAMES.index('RightShoulder')]] + scaled_shoulder1_to_shoulder2_pos_delta
    scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftForeArm'), bvh_cfg.BVH_JOINT_NAMES.index('RightForeArm')]] = scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
            'LeftArm'), bvh_cfg.BVH_JOINT_NAMES.index('RightArm')]] + scaled_shoulder2_to_elbow_pos_delta
    scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftHand'), bvh_cfg.BVH_JOINT_NAMES.index('RightHand')]] = scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
            'LeftForeArm'), bvh_cfg.BVH_JOINT_NAMES.index('RightForeArm')]] + scaled_elbow_to_hand_pos_delta

    scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftFingerBase'), bvh_cfg.BVH_JOINT_NAMES.index('RightFingerBase')]] = scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
            'LeftHand'), bvh_cfg.BVH_JOINT_NAMES.index('RightHand')]] + scaled_hand_to_finger1_pos_delta
    scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index('LeftHandIndex1'), bvh_cfg.BVH_JOINT_NAMES.index('RightHandIndex1')]] = scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftFingerBase'), bvh_cfg.BVH_JOINT_NAMES.index('RightFingerBase')]] + scaled_finger1_to_finger2_pos_delta
    scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index('LThumb'), bvh_cfg.BVH_JOINT_NAMES.index('RThumb')]] = scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftHandIndex1'), bvh_cfg.BVH_JOINT_NAMES.index('RightHandIndex1')]] + scaled_finger2_to_thumb_pos_delta

    # scale lower body (leg)
    scaled_hips_to_crotch_pos_delta = ref_hips_to_crotch_pos_delta * \
        khr_cfg.HIPS_TO_CROTCH_SCALE
    scaled_crotch_to_knee_pos_delta = ref_crotch_to_knee_pos_delta * \
        khr_cfg.CROTCH_TO_KNEE_SCALE
    scaled_knee_to_foot_pos_delta = ref_knee_to_foot_pos_delta * \
        khr_cfg.KNEE_TO_FOOT_SCALE
    scaled_foot_to_toe_pos_delta = ref_foot_to_toe_pos_delta * khr_cfg.FOOT_TO_TOE_SCALE

    scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftUpLeg'), bvh_cfg.BVH_JOINT_NAMES.index('RightUpLeg')]] = ref_hips_pos + scaled_hips_to_crotch_pos_delta
    scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftLeg'), bvh_cfg.BVH_JOINT_NAMES.index('RightLeg')]] = scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
            'LeftUpLeg'), bvh_cfg.BVH_JOINT_NAMES.index('RightUpLeg')]] + scaled_crotch_to_knee_pos_delta
    scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftFoot'), bvh_cfg.BVH_JOINT_NAMES.index('RightFoot')]] = scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
            'LeftLeg'), bvh_cfg.BVH_JOINT_NAMES.index('RightLeg')]] + scaled_knee_to_foot_pos_delta

    scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftToeBase'), bvh_cfg.BVH_JOINT_NAMES.index('RightToeBase')]] = scaled_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
            'LeftFoot'), bvh_cfg.BVH_JOINT_NAMES.index('RightFoot')]] + scaled_foot_to_toe_pos_delta

    return scaled_joint_pos


def retarget_pose(robot, ref_joint_pos):
    """
    robot: the robot need be retargeted
    ref_joint_pos: one frame data of global joint position [J+N, 3]
    """

    robot_joint_indices = get_robot_joint_indices(robot)
    robot_num_joints = len(robot_joint_indices)

    non_fixed_joint_indices = get_non_fixed_joint_indices(robot)
    joint_lower_bound, joint_upper_bound, joint_limit_range = get_joint_limits(
        robot)
    joint_lower_limit, joint_upper_limit, joint_limit_range = get_joint_limits2(
        robot)

    # align root pose by cross product each frame
    root_pos, root_quat = retarget_root_pose(ref_joint_pos)
    pybullet.resetBasePositionAndOrientation(robot, root_pos, root_quat)

    # reference joint position from frame
    scaled_joint_pos = scale_ref_pos(robot, ref_joint_pos)

    # set target joint
    target_neck_indices = [
        bvh_cfg.BVH_JOINT_NAMES.index('Hips'),
        bvh_cfg.BVH_JOINT_NAMES.index('Neck')
    ]

    target_elbow_indices = [
        bvh_cfg.BVH_JOINT_NAMES.index('LeftForeArm'),
        bvh_cfg.BVH_JOINT_NAMES.index('RightForeArm'),
    ]

    target_hand_indices = [
        bvh_cfg.BVH_JOINT_NAMES.index('LeftHand'),
        bvh_cfg.BVH_JOINT_NAMES.index('RightHand')
    ]

    target_foot_indices = [
        bvh_cfg.BVH_JOINT_NAMES.index('LeftLeg'),
        bvh_cfg.BVH_JOINT_NAMES.index('RightLeg'),
        bvh_cfg.BVH_JOINT_NAMES.index('LeftFoot'),
        bvh_cfg.BVH_JOINT_NAMES.index('RightFoot')
    ]

    # non_fixed_joint_indices = get_non_fixed_joint_indices(robot)
    # for k in range(len(non_fixed_joint_indices)):
    #     pybullet.resetJointState(
    #         robot, non_fixed_joint_indices[k], chest_joint_pose[k], 0.)

    target_neck_position = scaled_joint_pos[target_neck_indices]
    target_elbow_position = scaled_joint_pos[target_elbow_indices]
    target_hand_position = scaled_joint_pos[target_hand_indices]
    target_foot_position = scaled_joint_pos[target_foot_indices]

    target_elbow_position
    # target_joint_indices = target_foot_indices
    target_joint_indices = target_elbow_indices + \
        target_hand_indices + target_foot_indices
    # target_joint_indices = target_neck_indices + \
    #     target_hand_indices + target_foot_indices

    target_robot_joint_indices = [
        # spine target
        # robot_joint_indices['HIP_VIRTUAL_JOINT'], robot_joint_indices['HEAD_JOINT0'],
        # arm target
        # robot_joint_indices['LARM_VIRTUAL_JOINT2'], robot_joint_indices['RARM_VIRTUAL_JOINT2'],
        robot_joint_indices['LARM_JOINT2'], robot_joint_indices['RARM_JOINT2'],
        robot_joint_indices['LHAND_VIRTUAL_JOINT'], robot_joint_indices['RHAND_VIRTUAL_JOINT'],
        # leg target
        robot_joint_indices['LLEG_VIRTUAL_JOINT2'], robot_joint_indices['RLEG_VIRTUAL_JOINT2'],
        robot_joint_indices['LLEG_VIRTUAL_JOINT3'], robot_joint_indices['RLEG_VIRTUAL_JOINT3']
    ]

    target_joint_position = np.vstack(
        [
            # target_neck_position,
            target_elbow_position,
            target_hand_position,
            target_foot_position
        ])

    DEFAULT_JOINT_POS = np.array(
        joint_lower_limit) + np.array(joint_upper_limit) / 2.

    target_joint_pose = pybullet.calculateInverseKinematics2(robot,
                                                             endEffectorLinkIndices=target_robot_joint_indices,
                                                             targetPositions=target_joint_position,
                                                             jointDamping=khr_cfg.JOINT_DAMPING*robot_num_joints,
                                                             lowerLimits=joint_lower_limit,
                                                             upperLimits=joint_upper_limit,
                                                             jointRanges=joint_limit_range,
                                                             restPoses=DEFAULT_JOINT_POS
                                                             )

    # root_pos : (3,) root_quat : (4,) joint_pose : (J,) end_pos_local (4 * 3,), base_lin_vel_local (3, ) base_ang_vel_local(3.) joint_vel (J,), end_vel_local(4 * 3)
    pose = np.concatenate([root_pos, root_quat, target_joint_pose])
    return pose, scaled_joint_pos


def calculate_diff(f, dt):
    dfdt = np.zeros_like(f)
    df = f[1:, :] - f[0:-1, :]
    dfdt[:-1, :] = df / dt
    dfdt[-1, :] = dfdt[-2, :]

    return dfdt


#     return dfdt =


# def reorder_quat(quat, order='wxyz'):
#     _quat = np.zeros_like(quat)
#     if order == 'wxyz':
#         _quat[0] = quat[3]
#         _quat[1] = quat[0]
#         _quat[2] = quat[1]
#         _quat[3] = quat[2]
#     elif order == 'xyzw':
#         _quat[0] = quat[1]
#         _quat[1] = quat[2]
#         _quat[2] = quat[3]
#         _quat[3] = quat[0]
#     else:
#         raise NotImplementedError
#     return _quat

# def quat_rotate_inv(quat, vec):
#     """
#     rotation of vector by inverse of queaternion
#     """

#     quat_vec


def retarget_motion(robot, ref_joint_pos):
    num_frames = ref_joint_pos.shape[0]
    scaled_joint_pos_frames = np.zeros_like(ref_joint_pos)
    for f in range(num_frames):
        ref_joint_position = ref_joint_pos[f, ...]  # shape:[J+N,3]
        ref_joint_position = pre_process_ref_joint_pos(ref_joint_position)

        # ref_joint_position : right-handed, Z_UP_AXIS and scaled to robot
        curr_pose, scaled_joint_pos = retarget_pose(
            robot, ref_joint_position)
        set_pose(robot, curr_pose)  # for visualization
        # time.sleep(0.02)
        scaled_joint_pos_frames[f] = scaled_joint_pos

        if f == 0:
            pose_size = curr_pose.shape[-1]
            new_frames = np.zeros([num_frames, pose_size])
        new_frames[f] = curr_pose

    print("%d frames" % num_frames)
    new_frames[:, 0:2] -= new_frames[0, 0:2]
    return new_frames, scaled_joint_pos_frames


def set_robot_joint_marker(robot, marker_ids):

    num_joints = pybullet.getNumJoints(robot)
    assert(num_joints == len(marker_ids))

    robot_joint_pos = np.array(pybullet.getLinkStates(robot, list(
        range(num_joints)), computeForwardKinematics=True))[:, 4]

    for i in range(len(marker_ids)):
        curr_id = marker_ids[i]
        curr_pos = robot_joint_pos[i]
        pybullet.resetBasePositionAndOrientation(
            curr_id, curr_pos, np.array([0, 0, 0, 1]))


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


def get_robot_joint_indices(robot):
    robot_joint_indices = {}
    for i in range(pybullet.getNumJoints(robot)):
        joint_name = str(pybullet.getJointInfo(robot, i)[1], 'utf-8')
        # print(pybullet.getJointInfo(robot, i))
        robot_joint_indices[joint_name] = i

    return robot_joint_indices


def output_motion(frames, out_filename):
    with open(out_filename, "w") as f:
        f.write("{\n")
        f.write("\"LoopMode\": \"Wrap\",\n")
        f.write("\"FrameDuration\": " + str(bvh_cfg.FRAME_DURATION) + ",\n")
        f.write("\"EnableCycleOffsetPosition\": true,\n")
        f.write("\"EnableCycleOffsetRotation\": true,\n")
        f.write("\"MotionWeight\": 1.0,\n")
        f.write("\n")

        f.write("\"Frames\":\n")

        f.write("[")
        for i in range(frames.shape[0]):
            curr_frame = frames[i]

            if i != 0:
                f.write(",")
            f.write("\n  [")

            for j in range(frames.shape[1]):
                curr_val = curr_frame[j]
                if j != 0:
                    f.write(", ")
                f.write("%.5f" % curr_val)

            f.write("]")

        f.write("\n]")
        f.write("\n}")

    return


def main():
    # build world
    robot, ground = build_world()
    num_joints = pybullet.getNumJoints(robot)
    robot_joint_indices = get_robot_joint_indices(robot)
    print("robot's joint :", robot_joint_indices)

    print("robot's non-fixed-joint :", get_non_fixed_joint_indices(robot))

    # create marker to display reference motion
    num_bvh_joints = len(bvh_cfg.BVH_JOINT_NAMES)
    # ref_bvh_joint_marker_ids = build_markers(
    #     num_bvh_joints, col=[0., 1., 0., 1.], size=0.005)
    robot_joint_marker_ids = build_markers(
        num_joints, col=[0., 0., 1., 1.], size=0.005)
    scaled_bvh_joint_pos_marker_ids = build_markers(
        num_bvh_joints, col=[0.9, 0., 0.7, 1.], size=0.01)

    # load global joint position from BVH
    bvh_file_list = bvh_cfg.FILE_LIST
    for i in range(len(bvh_file_list)):
        bvh_file = bvh_file_list[i]
        file_path = bvh_cfg.FILE_DIR + bvh_file

        # retarget motion using scaled joint global position
        # ref_joint_pos_frames : (F : frames, J : number of bvh joints, 3 : global position)
        # retarget_frames, scaled_joint_pos_frames : (F : frames, J : number of robot joints, )
        ref_joint_pos_frames, bvh_joint_names = get_joint_global_pos(file_path)
        retarget_frames, scaled_joint_pos_frames = retarget_motion(
            robot, ref_joint_pos_frames)

        dof_pos = retarget_frames[:, 7:]
        dof_vel = calculate_diff(dof_pos, dt=bvh_cfg.FRAME_DURATION)

        f = 0  # frame count for display
        num_frames = ref_joint_pos_frames.shape[0]  # frames for motion

        #
        ref_joint_pos = []
        for frame in range(num_frames):
            ref_joint_pos.append(
                pre_process_ref_joint_pos(ref_joint_pos_frames[frame]))

        motion_num = bvh_cfg.FILE_LIST[i].split('.')[0]
        save_path = bvh_cfg.OUT_FILE_DIR + motion_num

        ref_joint_pos = np.array(ref_joint_pos)
        np.savez(save_path,
                 retarget_frames=retarget_frames,
                 ref_joint_pos=ref_joint_pos,
                 robot_joint_indices=robot_joint_indices,
                 non_fixed_joint_indices=get_non_fixed_joint_indices(robot),
                 frame_duration=bvh_cfg.FRAME_DURATION,
                 allow_pickle=True)

        # output_file_name = bvh_cfg.OUT_FILE_DIR + \
        #     bvh_file.split(".")[0] + '.txt'
        # output_motion(retarget_frames, output_file_name)

        while True:
            # pybullet.resetSimulation()
            time_start = time.time()

            f_idx = f % num_frames
            pose = retarget_frames[f_idx]
            set_pose(robot, pose)

            # set_marker_pos(ref_joint_pos[f_idx], ref_bvh_joint_marker_ids)
            set_robot_joint_marker(robot, robot_joint_marker_ids)
            set_marker_pos(
                scaled_joint_pos_frames[f_idx], scaled_bvh_joint_pos_marker_ids)

            time_end = time.time()
            sleep_dur = bvh_cfg.FRAME_DURATION - (time_end - time_start)
            sleep_dur = max(0, sleep_dur)

            time.sleep(sleep_dur)
            f += 1
            # input()

        pybullet.disconnect()

        return


if __name__ == '__main__':
    main()
