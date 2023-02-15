import math
import os

from isaacgym import gymapi, gymutil
from isaacgymenvs import ISAAC_GYM_ROOT_DIR
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import scipy.ndimage.filters as filters
from scipy import stats
from utils import rotation3d

import torch
import numpy as np

import config.bvh_cfg.bvh_cmu_config as bvh_cfg

"""
urdf model with virtual joint to real model
"""

FILTER = True
Y_SYMMETRIC = True
Z_STABLE = True
X_POS_SCALE = 0.8
CLIPPING = 10
# Z_OFFSET = 0.067
Z_OFFSET = 0.073
# Z_OFFSET = 0.1

# KHR_URDF = 'khr_set_limit_joint_w_virtual_joint.urdf'
# KHR_URDF = 'khr.urdf'
URDF_MODEL = 'khr_set_limit_joint.urdf'
# URDF_MODEL = 'khr_set_limit_joint_w_small_foot_ver1.urdf'

MOTION_NUM = "07_08"
MOTION_FILE_PATH = os.path.join("./result/output", MOTION_NUM+".npz")
OUTPUT_FILE_PATH = os.path.join(
    "./result/output/isaac_motion", MOTION_NUM + ".npz")

#----------------------------------------function----------------------------------------#


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


def Vec3toTuple(vec3):
    return (vec3.x, vec3.y, vec3.z)


def compute_linear_velocity(f, dt, level=6, gaussian_filter=True):
    dfdt = np.zeros_like(f)
    df = f[1:, :] - f[0:-1, :]
    dfdt[:-1, :] = df / dt
    dfdt[-1, :] = dfdt[-2, :]

    if gaussian_filter:
        dfdt = filters.gaussian_filter1d(dfdt, level, axis=-2, mode="nearest")

    return dfdt


def compute_angular_velocity(r, time_delta: float, gaussian_filter=True):
    # assume the second last dimension is the time axis
    diff_quat_data = rotation3d.quat_identity_like(r)
    diff_quat_data[:-1] = rotation3d.quat_mul_norm(
        r[1:, :], rotation3d.quat_inverse(r[:-1, :])
    )
    diff_quat_data[-1] = diff_quat_data[-2]
    diff_angle, diff_axis = rotation3d.quat_angle_axis(diff_quat_data)
    angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta

    if gaussian_filter:
        angular_velocity = torch.from_numpy(
            filters.gaussian_filter1d(
                angular_velocity.numpy(), 2, axis=-2, mode="nearest"
            ),
        )
    return angular_velocity


def remove_bias(biased_data, remove_mean=True):
    x = np.arange(biased_data.shape[0])
    linear_regression = stats.linregress(x, biased_data)
    if remove_mean:
        mean = 0.
    else:
        mean = np.mean(biased_data)
    result = biased_data - (linear_regression.intercept +
                            linear_regression.slope * x) + mean
    return result


#----------------------------------------function----------------------------------------#


def main(urdf_model,
         motion_num,
         motion_file_path,
         output_file_path):
    motion_data = np.load(motion_file_path, allow_pickle=True)
    retarget_frames = motion_data["retarget_frames"]
    frame_duration = motion_data["frame_duration"]
    non_fixed_joint_indices = motion_data["non_fixed_joint_indices"]

    dof_num = len(non_fixed_joint_indices)
    fps = int(1.0 / frame_duration)
    num_frames = retarget_frames.shape[0]

    """
    pybullet retarget frames :
    root pos x,y,z (m)
    root quat x,y,z,w (quat)
    dof_pos : angle (rad)
    """

    base_pos = retarget_frames[:, 0:3]
    base_quat = retarget_frames[:, 3:7]
    dof_pos = retarget_frames[:, 7:7+dof_num]

    if X_POS_SCALE is not None:
        base_pos[:, 0] = base_pos[:, 0] * X_POS_SCALE

    if FILTER:
        for i in range(dof_pos.shape[-1]):
            dof_pos[:, i] = filters.gaussian_filter1d(
                dof_pos[:, i], 2, axis=-1, mode="nearest")

    """
    reorder joint order from pybullet to isaac
    pybullet joint order : LARM, RARM, LLEG, RLEG

    if khr_set_limit_joint_w_virtual_joint.urdf is selected :
    isaac joint order : LLEG, RLEG, LARM, RARM
    elif khr.urdf :
    isaac joint order : LARM, LLEG, RARM, RLEG
    """

    dof_larm = dof_pos[:, 0:3]
    dof_rarm = dof_pos[:, 3:6]
    dof_lleg = dof_pos[:, 6:11]
    dof_rleg = dof_pos[:, 11:16]

    if Y_SYMMETRIC:
        base_pos[:, 1] = remove_bias(biased_data=base_pos[:, 1])
        # yaw = remove_bias(biased_data=yaw, remove_mean=False)

    if Z_STABLE:
        base_pos[:, 2] = remove_bias(
            biased_data=base_pos[:, 2], remove_mean=False)

    if Z_OFFSET:
        base_pos[:, 2] += Z_OFFSET

    if FILTER:
        for i in range(base_quat.shape[-1]):
            base_quat[:, i] = filters.gaussian_filter1d(
                base_quat[:, i], 2, axis=-1, mode="nearest")
        base_quat = quat_unit(to_torch(base_quat, device='cpu')).numpy()

    if "virtual" in urdf_model:
        dof_pos = np.hstack([dof_lleg, dof_rleg, dof_larm, dof_rarm])
    else:
        dof_pos = np.hstack([dof_larm, dof_lleg, dof_rarm, dof_rleg])

    # convert lin_vel and ang_vel to local frame
    base_quat_world = to_torch(base_quat, device='cpu')

    base_lin_vel = compute_linear_velocity(
        base_pos, dt=frame_duration, level=1, gaussian_filter=FILTER)
    base_ang_vel = compute_angular_velocity(
        base_quat_world, time_delta=frame_duration, gaussian_filter=FILTER).numpy()

    end_effector_pos = np.zeros((num_frames, 3 * 4))

    # root_states in world
    root_states = np.hstack(
        [base_pos, base_quat, base_lin_vel, base_ang_vel])

    # simple asset descriptor for selecting from a list
    class AssetDesc:
        def __init__(self, file_name, flip_visual_attachments=False):
            self.file_name = file_name
            self.flip_visual_attachments = flip_visual_attachments

    asset_descriptors = [
        AssetDesc(os.path.join("urdf/khr/", urdf_model), False),
    ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="Motion Retarget: motion retarget scripts for robots",
        custom_parameters=[
            {"name": "--asset_id", "type": int, "default": 0,
                "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)},
            {"name": "--speed_scale", "type": float,
                "default": 1.0, "help": "Animation speed scale"},
            {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])

    if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
        print("*** Invalid asset_id specified.  Valid range is 0 to %d" %
              (len(asset_descriptors) - 1))
        quit()

    # initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = dt = frame_duration
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0., 0., -9.8)
    if args.physics_engine == gymapi.SIM_FLEX:
        pass
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id,
                         args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0., 0., 1.)
    gym.add_ground(sim, plane_params)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    # load asset
    asset_root = os.path.join(ISAAC_GYM_ROOT_DIR, "assets")
    asset_file = asset_descriptors[args.asset_id].file_name

    asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
    asset_options.use_mesh_materials = True
    asset_options.collapse_fixed_joints = True

    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # get array of DOF names
    dof_names = gym.get_asset_dof_names(asset)
    print('DOF NAMES')
    print(dof_names)

    # get array of DOF properties
    dof_props = gym.get_asset_dof_properties(asset)

    # create an array of DOF states that will be used to update the actors
    num_dofs = gym.get_asset_dof_count(asset)
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    initial_state = np.copy(
        gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    # get list of DOF types
    dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

    # get the position slice of the DOF state array
    dof_positions = dof_states['pos']

    # get the limit-related slices of the DOF properties array
    stiffnesses = dof_props['stiffness']
    dampings = dof_props['damping']
    armatures = dof_props['armature']
    has_limits = dof_props['hasLimits']
    lower_limits = dof_props['lower']
    upper_limits = dof_props['upper']

    # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
    defaults = np.zeros(num_dofs)
    speeds = np.zeros(num_dofs)
    for i in range(num_dofs):
        if has_limits[i]:
            if dof_types[i] == gymapi.DOF_ROTATION:
                lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
                upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
            # make sure our default position is in range
            if lower_limits[i] > 0.0:
                defaults[i] = lower_limits[i]
            elif upper_limits[i] < 0.0:
                defaults[i] = upper_limits[i]
        else:
            # set reasonable animation limits for unlimited joints
            if dof_types[i] == gymapi.DOF_ROTATION:
                # unlimited revolute joint
                lower_limits[i] = -math.pi
                upper_limits[i] = math.pi
            elif dof_types[i] == gymapi.DOF_TRANSLATION:
                # unlimited prismatic joint
                lower_limits[i] = -1.0
                upper_limits[i] = 1.0
        # set DOF position to default
        dof_positions[i] = defaults[i]

    # Print DOF properties
    for i in range(num_dofs):
        print("DOF %d" % i)
        print("  Name:     '%s'" % dof_names[i])
        print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
        print("  Stiffness:  %r" % stiffnesses[i])
        print("  Damping:  %r" % dampings[i])
        print("  Armature:  %r" % armatures[i])
        print("  Limited?  %r" % has_limits[i])
        if has_limits[i]:
            print("    Lower   %f" % lower_limits[i])
            print("    Upper   %f" % upper_limits[i])

    dof_pos = np.clip(dof_pos, a_min=np.array(
        lower_limits), a_max=np.array(upper_limits))
    dof_vel = compute_linear_velocity(
        dof_pos, dt=frame_duration, level=1, gaussian_filter=FILTER)

    # set up the env grid
    num_envs = 36
    num_per_row = 6
    spacing = 0.5
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # position the camera
    cam_pos = gymapi.Vec3(6.0, 6.0, 2)
    cam_target = gymapi.Vec3(3.0, 0.0, 0.3)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # cache useful handles
    envs = []
    actor_handles = []

    print("Creating %d environments" % num_envs)
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        # pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -0.5 * math.pi)
        # pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
        actor_handles.append(actor_handle)

        # set default DOF positions
        gym.set_actor_dof_states(
            env, actor_handle, dof_states, gymapi.STATE_ALL)

    # initialize animation state
    # hand_names = ["LARM_LINK2", "RARM_LINK2"]
    # foot_names = ["LLEG_LINK4", "RLEG_LINK4"]
    # LINK_NAMES = gym.get_actor_rigid_body_names(envs[0], actor_handles[0])
    LINK_NAMES = gym.get_asset_rigid_body_names(asset)

    LARM_END_EFFECTOR_INDEX = gym.find_actor_rigid_body_handle(
        envs[0], actor_handles[0], "LARM_LINK2")
    RARM_END_EFFECTOR_INDEX = gym.find_actor_rigid_body_handle(
        envs[0], actor_handles[0], "RARM_LINK2")
    LLEG_END_EFFECTOR_INDEX = gym.find_actor_rigid_body_handle(
        envs[0], actor_handles[0], "LLEG_LINK4")
    RLEG_END_EFFECTOR_INDEX = gym.find_actor_rigid_body_handle(
        envs[0], actor_handles[0], "RLEG_LINK4")

    print("LINK NAMES")
    print(LINK_NAMES)

    print("LARM EFFECTOR INDEX")
    print(LARM_END_EFFECTOR_INDEX)
    print("RARM EFFECTOR INDEX")
    print(RARM_END_EFFECTOR_INDEX)
    print("LLEG EFFECTOR INDEX")
    print(LLEG_END_EFFECTOR_INDEX)
    print("RLEG EFFECTOR INDEX")
    print(RLEG_END_EFFECTOR_INDEX)

    """
    LINK NAMES
    ['BODY', 'LARM_LINK0', 'LARM_LINK1', 'LARM_LINK2', 'LLEG_LINK0', 'LLEG_LINK1', 'LLEG_LINK2', 'LLEG_LINK3', 'LLEG_LINK4', 'RARM_LINK0', 'RARM_LINK1', 'RARM_LINK2', 'RLEG_LINK0', 'RLEG_LINK1', 'RLEG_LINK2', 'RLEG_LINK3', 'RLEG_LINK4']
    17
    DOF NAMES
    ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LLEG_JOINT0', 'LLEG_JOINT1', 'LLEG_JOINT2', 'LLEG_JOINT3', 'LLEG_JOINT4', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RLEG_JOINT0', 'RLEG_JOINT1', 'RLEG_JOINT2', 'RLEG_JOINT3', 'RLEG_JOINT4']
    16
    """

    frame = 0

    while not gym.query_viewer_has_closed(viewer):

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if args.show_axis:
            gym.clear_lines(viewer)

        dof_positions[:] = dof_pos[frame % num_frames]
        root_positions = root_states[frame % num_frames]

        root_pos = gymapi.Vec3(*root_positions[0:3].tolist())
        root_quat = gymapi.Quat(*root_positions[3:7].tolist())
        root_lin_vel = gymapi.Vec3(*root_positions[7:10].tolist())
        root_ang_vel = gymapi.Vec3(*root_positions[10:13].tolist())

        # clone actor state in all of the environments
        for i in range(num_envs):

            state = gym.get_actor_rigid_body_states(
                envs[i], actor_handles[i], gymapi.STATE_NONE)  # 'pose',
            state['pose']['p'].fill((root_pos.x, root_pos.y, root_pos.z))
            state['pose']['r'].fill(
                (root_quat.x, root_quat.y, root_quat.z, root_quat.w))
            state['vel']['linear'].fill(
                (root_lin_vel.x, root_lin_vel.y, root_lin_vel.z))
            state['vel']['angular'].fill(
                (root_ang_vel.x, root_ang_vel.y, root_ang_vel.z))

            # TODO : debugging
            gym.set_actor_rigid_body_states(
                envs[i], actor_handles[i], state, gymapi.STATE_ALL)

            # gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

            gym.set_actor_dof_states(
                envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

        state_collected = gym.get_actor_rigid_body_states(
            envs[0], actor_handles[0], gymapi.STATE_ALL)

        LARM_END_EFFECTOR_POS = list(
            state_collected[LARM_END_EFFECTOR_INDEX]['pose']['p'])
        RARM_END_EFFECTOR_POS = list(
            state_collected[RARM_END_EFFECTOR_INDEX]['pose']['p'])
        LLEG_LINK4_POS = list(
            state_collected[LLEG_END_EFFECTOR_INDEX]['pose']['p'])
        RLEG_LINK4_POS = list(
            state_collected[RLEG_END_EFFECTOR_INDEX]['pose']['p'])

        if "virtual" in urdf_model:
            end_effector_pos_world = np.array(
                LLEG_LINK4_POS + RLEG_LINK4_POS + LARM_END_EFFECTOR_POS + RARM_END_EFFECTOR_POS)
        else:
            end_effector_pos_world = np.array(
                [LARM_END_EFFECTOR_POS + LLEG_LINK4_POS + RARM_END_EFFECTOR_POS + RLEG_LINK4_POS])

        end_effector_pos[frame % num_frames] = end_effector_pos_world

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
        frame += 1

    key_pos = end_effector_pos.reshape(-1, 4, 3)

    processed_motion_data = np.hstack(
        [base_pos, base_quat, dof_pos, base_lin_vel, base_ang_vel, dof_vel, end_effector_pos])

    processed_motion_data = processed_motion_data[CLIPPING:-CLIPPING]

    if FILTER:
        np.save('./result/temp/filtered_motion_data.npy', processed_motion_data)
        np.savez(output_file_path,
                 root_pos=base_pos,
                 root_rot=base_quat,
                 dof_pos=dof_pos,
                 root_lin_vel=base_lin_vel,
                 root_ang_vel=base_ang_vel,
                 dof_vel=dof_vel,
                 key_pos=key_pos,
                 motion_fps=fps)
    else:
        np.save('./result/temp/raw_motion_data.npy', processed_motion_data)

    print("done")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main(urdf_model=URDF_MODEL,
         motion_num=MOTION_NUM,
         motion_file_path=MOTION_FILE_PATH,
         output_file_path=OUTPUT_FILE_PATH)
