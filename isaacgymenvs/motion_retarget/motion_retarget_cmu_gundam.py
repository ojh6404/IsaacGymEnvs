import time
import numpy as np
import numpy.core.umath_tests as ut
from utils import BVH, Animation
from utils import pose3d

import pybullet
import pybullet_data as pd
from pybullet_utils import transformations

import config.bvh_cfg.bvh_cmu_config as bvh_cfg
import config.robot_cfg.gundam_retarget_config_cmu as gundam_cfg


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


def calc_end_eff_pos(animation: Animation):
    """
     calculate the end-site global position
     append it into the end of joint position list
    -------
    """
    joint_global_transform = animation.transforms_global(animation)
    num_frames = joint_global_transform.shape[0]
    LH_joint_transform = joint_global_transform[:, bvh_cfg.LH_JOINT_IDX, :, :]
    RH_joint_transform = joint_global_transform[:, bvh_cfg.RH_JOINT_IDX, :, :]
    LF_joint_transform = joint_global_transform[:, bvh_cfg.LF_JOINT_IDX, :, :]
    RF_joint_transform = joint_global_transform[:, bvh_cfg.RF_JOINT_IDX, :, :]

    LH_end_global_position = ut.matrix_multiply(
        LH_joint_transform, bvh_cfg.LH_END_OFFSET).reshape(num_frames, 1, 4)[:, :, 3]
    RH_end_global_position = ut.matrix_multiply(
        RH_joint_transform, bvh_cfg.RH_END_OFFSET).reshape(num_frames, 1, 4)[:, :, 3]
    LF_end_global_position = ut.matrix_multiply(
        LF_joint_transform, bvh_cfg.LF_END_OFFSET).reshape(num_frames, 1, 4)[:, :, 3]
    RF_end_global_position = ut.matrix_multiply(
        RF_joint_transform, bvh_cfg.RF_END_OFFSET).reshape(num_frames, 1, 4)[:, :, 3]

    end_effector_position = np.concatenate(
        (LH_end_global_position, RH_end_global_position, LF_end_global_position, RF_end_global_position), axis=1)
    return end_effector_position


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

    if bvh_cfg.CAL_END_EFFECTOR:
        end_site_position = calc_end_eff_pos(anim)
        joint_position = np.concatenate(
            (joint_position, end_site_position), axis=1)
    joint_position *= bvh_cfg.POSITION_SCALING
    joint_position = set_joint_pos_origin(joint_position)
    return joint_position, joint_names


# ----------------------------------------------------------------------------------------------------------------------


def build_markers(num_markers):
    marker_radius = 0.005
    markers_handle = []
    for i in range(num_markers):
        if (bvh_cfg.BVH_JOINT_NAMES[i] == 'Hips') or (bvh_cfg.BVH_JOINT_NAMES[i] == 'Spine')\
                or (bvh_cfg.BVH_JOINT_NAMES[i] == 'Spine1') or (bvh_cfg.BVH_JOINT_NAMES[i] == 'Neck'):
            col = [0, 1, 0, 1]
        elif 'Left' in bvh_cfg.BVH_JOINT_NAMES[i]:
            col = [0.9, 0, 0.7, 1]
        elif 'Right' in bvh_cfg.BVH_JOINT_NAMES[i]:
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


def pre_process_ref_joint_pos(ref_joint_pos):
    """
    1. align the coordinate: most bvh file use right-hand and y-up axis, while gundam urdf and pybullet use z-up with right-hand coordinate
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
            curr_pos = curr_pos * gundam_cfg.REF_POS_SCALE
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
    root_height_scale = gundam_cfg.GUNDAM_ROOT_HEIGHT / bvh_cfg.BVH_ROOT_HEIGHT
    # root_position = root_position * root_height_scale  # TODO: scale z or scale x, y and z?
    # root_position[-1] = root_position[-1] + gundam_cfg.GUNDAM_ROOT_HEIGHT
    return root_position, root_quaternion


def get_non_fixed_joint_indices(robot):
    num_joints = pybullet.getNumJoints(robot)
    non_fixed_joint_indices = []
    for i in range(num_joints):
        joint_type = pybullet.getJointInfo(robot, i)[2]
        if joint_type is not 4:
            non_fixed_joint_indices.append(i)
    return non_fixed_joint_indices


def retarget_pose(robot, ref_joint_pos, robot_joint_indices):
    """
    robot: the robot need be retargeted
    ref_joint_pos: one frame data of global joint position [J+N, 3]
    """
    robot_num_joints = len(robot_joint_indices)
    # align root pose by cross product each frame
    root_pos, root_quater = retarget_root_pose(ref_joint_pos)
    pybullet.resetBasePositionAndOrientation(robot, root_pos, root_quater)

    # retarget chest joint
    # ref_root_joint_position = ref_joint_pos[0]
    # ref_neck_joint_position = ref_joint_pos[bvh_cfg.BVH_JOINT_NAMES.index(
    #     'Neck')]
    # ref_chest_delta = ref_neck_joint_position - ref_root_joint_position

    # chest_scale = gundam_cfg.GUNDAM_CHEST_LENGTH / bvh_cfg.BVH_CHESET_LENGTH

    # # gundam_chest_delta = ref_chest_delta * chest_scale  #TODO: only z or x,y,z??
    # gundam_chest_delta = ref_chest_delta
    # target_chest2_position = root_pos + gundam_chest_delta
    # chest_joint_pose = pybullet.calculateInverseKinematics(robot,
    #                                                        endEffectorLinkIndex=robot_joint_indices['CHEST_JOINT2'],
    #                                                        targetPosition=target_chest2_position,
    #                                                        jointDamping=gundam_cfg.JOINT_DAMPING*robot_num_joints)

    # chest_joint_pose = np.array(chest_joint_pose)
    # chest_joint_pose[0] = 0.
    # chest_joint_pose[1] = 0.
    # # print('chest_joint_pose',chest_joint_pose)
    # non_fixed_joint_indices = get_non_fixed_joint_indices(robot)
    # for k in range(len(non_fixed_joint_indices)):
    #     pybullet.resetJointState(
    #         robot, non_fixed_joint_indices[k], chest_joint_pose[k], 0.)

    # retarget hand
    # ref_hand_position = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
    #     'LeftHand'), bvh_cfg.BVH_JOINT_NAMES.index('RightHand')]]
    # ref_arm_position = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
    #     'LeftArm'), bvh_cfg.BVH_JOINT_NAMES.index('RightArm')]]
    # ref_hand_delta = ref_hand_position - ref_arm_position

    # hand_scale = gundam_cfg.GUNDAM_HAND_LENGTH / bvh_cfg.BVH_HAND_LENGTH

    # gundam_hand_delta = ref_hand_delta * hand_scale
    # gundam_hand_delta = ref_hand_delta
    # gundam_larm_link2_position = np.array(pybullet.getLinkState(
    #     robot, robot_joint_indices['larm_elbow_p'], computeForwardKinematics=True)[4])
    # gundam_rarm_link2_position = np.array(pybullet.getLinkState(
    #     robot, robot_joint_indices['rarm_elbow_p'], computeForwardKinematics=True)[4])
    # gundam_arm_link2_position = np.vstack(
    #     (gundam_larm_link2_position, gundam_rarm_link2_position))
    # target_hand_position = gundam_arm_link2_position + gundam_hand_delta
    # joint_lower_bound, joint_upper_bound, joint_limit_range = get_joint_limits(
    #     robot)
    # upper_body_joint_pose = pybullet.calculateInverseKinematics2(robot,
    #                                                              endEffectorLinkIndices=[
    #                                                                  robot_joint_indices['larm_gripper'], robot_joint_indices['rarm_gripper']],
    #                                                              targetPositions=target_hand_position,
    #                                                              jointDamping=gundam_cfg.JOINT_DAMPING*robot_num_joints
    #                                                              # lowerLimits=joint_lower_bound,
    #                                                              # upperLimits=joint_upper_bound,
    #                                                              # jointRanges=joint_limit_range,
    #                                                              # restPoses=16 *
    #                                                              # [0.017]
    #                                                              )

    non_fixed_joint_indices = get_non_fixed_joint_indices(robot)

    # for k in range(len(non_fixed_joint_indices)):
    #     pybullet.resetJointState(
    #         robot, non_fixed_joint_indices[k], upper_body_joint_pose[k], 0.)

    # retarget foot
    ref_foot_position = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftFoot'), bvh_cfg.BVH_JOINT_NAMES.index('RightFoot')]]
    ref_upper_leg_position = ref_joint_pos[[bvh_cfg.BVH_JOINT_NAMES.index(
        'LeftUpLeg'), bvh_cfg.BVH_JOINT_NAMES.index('RightUpLeg')]]
    ref_leg_delta = ref_foot_position - ref_upper_leg_position

    leg_scale = gundam_cfg.GUNDAM_LEG_LENGTH / \
        (bvh_cfg.BVH_ROOT_HEIGHT + 0.015)
    # gundam_foot_delta = ref_leg_delta * leg_scale
    gundam_foot_delta = ref_leg_delta
    gundam_lleg_link0_position = np.array(pybullet.getLinkState(
        robot, robot_joint_indices['lleg_crotch_p'], computeForwardKinematics=True)[4])
    gundam_rleg_link0_position = np.array(pybullet.getLinkState(
        robot, robot_joint_indices['rleg_crotch_p'], computeForwardKinematics=True)[4])
    gundam_leg_link0_position = np.vstack(
        (gundam_lleg_link0_position, gundam_rleg_link0_position))
    target_foot_position = gundam_leg_link0_position + gundam_foot_delta
    # target_foot_position[:, 2] = ref_foot_position[:, 2] # make the foot clearance more stable

    # TODO: add foot orientation in global
    target_joint_pose = pybullet.calculateInverseKinematics2(robot,
                                                             endEffectorLinkIndices=[
                                                                 robot_joint_indices['lleg_ankle_p'], robot_joint_indices['rleg_ankle_p']],
                                                             targetPositions=target_foot_position,
                                                             jointDamping=gundam_cfg.JOINT_DAMPING*robot_num_joints)
    # lowerLimits=joint_lower_bound,
    # upperLimits=joint_upper_bound,
    # jointRanges=joint_limit_range,
    # restPoses=chest_joint_pose)
    # print("target_joint_pose", target_joint_pose)

    # root_pos : (3,) root_quater : (4,) target_joint_pose : J
    pose = np.concatenate([root_pos, root_quater, target_joint_pose])
    return pose


def retarget_motion(robot, ref_joint_pos, robot_joint_indices):
    num_frames = ref_joint_pos.shape[0]
    print("retargeted file has %d frames" % num_frames)
    for f in range(num_frames):
        ref_joint_position = ref_joint_pos[f, ...]  # shape:[J+N,3]
        ref_joint_position = pre_process_ref_joint_pos(ref_joint_position)

        # ref_joint_position : right-handed, Z_UP_AXIS and scaled to robot
        curr_pose = retarget_pose(
            robot, ref_joint_position, robot_joint_indices)
        set_pose(robot, curr_pose)  # for visualization
        # time.sleep(0.02)

        if f == 0:
            pose_size = curr_pose.shape[-1]
            new_frames = np.zeros([num_frames, pose_size])
        new_frames[f] = curr_pose

    new_frames[:, 0:2] -= new_frames[0, 0:2]
    return new_frames


def main():
    # build world
    pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pd.getDataPath())
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, -9.8)
    ground = pybullet.loadURDF(
        gundam_cfg.GROUND_URDF_FILENAME, basePosition=[0., 0., 0.])

    # create actor
    robot = pybullet.loadURDF(gundam_cfg.ROBOT_URDF_FILENAME, basePosition=np.array(
        [0, 0, 12.5]), baseOrientation=np.array([0, 0, 0, 1]))
    num_joints = pybullet.getNumJoints(robot)
    robot_joint_indices = {}
    for i in range(num_joints):
        joint_name = str(pybullet.getJointInfo(robot, i)[1], 'utf-8')
        # print(pybullet.getJointInfo(robot, i))
        robot_joint_indices[joint_name] = i

    print(robot_joint_indices)

    print("num_joints_not_fixed")
    print(len(get_non_fixed_joint_indices(robot)))

    # create marker to display reference motion
    num_markers = len(bvh_cfg.BVH_JOINT_NAMES)
    marker_ids = build_markers(num_markers)

    # load global joint position from BVH file
    bvh_file_list = bvh_cfg.FILE_LIST
    for i in range(len(bvh_file_list)):
        bvh_file = bvh_file_list[i]
        file_path = bvh_cfg.FILE_DIR + bvh_file
        # (F, J+N, 3) J : joint_num, N : end_effector_num
        ref_joint_position, bvh_joint_names = get_joint_global_pos(file_path)

        retarget_frames = retarget_motion(
            robot, ref_joint_position, robot_joint_indices)

        f = 0
        num_frames = ref_joint_position.shape[0]

        ref_joint_pos = []

        for i in range(num_frames):
            ref_joint_pos.append(
                pre_process_ref_joint_pos(ref_joint_position[i]))

        ref_joint_pos = np.array(ref_joint_pos)

        # np.savez('./test', retarget_frames=retarget_frames,
        #          ref_joint_pos=ref_joint_pos)
        while True:
            time_start = time.time()

            f_idx = f % num_frames
            pose = retarget_frames[f_idx]
            set_pose(robot, pose)

            set_maker_pos(ref_joint_pos[f_idx], marker_ids)
            f += 1

            time_end = time.time()
            sleep_dur = bvh_cfg.FRAME_DURATION - (time_end - time_start)
            sleep_dur = max(0, sleep_dur)

            time.sleep(sleep_dur)

        pybullet.disconnect()

        return


if __name__ == '__main__':
    main()
