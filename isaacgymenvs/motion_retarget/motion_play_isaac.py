import math
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgymenvs import ISAAC_GYM_ROOT_DIR
import os

import config.bvh_cfg.bvh_cmu_config as bvh_cfg


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


def Vec3toTuple(vec3):
    return (vec3.x, vec3.y, vec3.z)


motion_file_path = "./result/temp/filtered_motion_data.npy"

retarget_frames = np.load(motion_file_path)
FRAME_DURATION = bvh_cfg.FRAME_DURATION

num_frames = retarget_frames.shape[0]

"""
reorder joint order from pybullet to isaac
pybullet joint order : LARM, RARM, LLEG, RLEG

if khr_set_limit_joint_w_virtual_joint.urdf is selected :
isaac joint order : LLEG, RLEG, LARM, RARM
elif khr.urdf :
isaac joint order : LARM, LLEG, RARM, RLEG
"""

KHR_URDF = 'khr.urdf'
# KHR_URDF = 'khr_set_limit_joint_w_virtual_joint.urdf'

base_pos = retarget_frames[:, 0:3]
base_quat = retarget_frames[:, 3:7]
dof_larm = retarget_frames[:, 7:10]
dof_lleg = retarget_frames[:, 10:15]
dof_rarm = retarget_frames[:, 15:18]
dof_rleg = retarget_frames[:, 18:23]


if KHR_URDF == 'khr.urdf':
    dof_pos = np.hstack([dof_larm, dof_lleg, dof_rarm, dof_rleg])
else:
    dof_pos = np.hstack([dof_lleg, dof_rleg, dof_larm, dof_rarm])

# root_states in world
root_states = np.hstack([base_pos, base_quat])


# simple asset descriptor for selecting from a list
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc(os.path.join("urdf/khr/", KHR_URDF), False),
]


# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
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
# sim_params.dt = dt = 1.0 / 60.0
sim_params.dt = dt = FRAME_DURATION
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
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

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
    # set speed depending on DOF type and range of motion
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * \
            clamp(2 * (upper_limits[i] - lower_limits[i]),
                  0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = args.speed_scale * \
            clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

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
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)


"""
khr_set_joint_limit_w_virtual_joint.urdf
['HIP_VIRTUAL_LINK', 'SPINE_VIRTUAL_LINK', 'LLEG_LINK0', 'LLEG_LINK1', 'LLEG_LINK2', 'LLEG_LINK3', 'LLEG_LINK4', 'LLEG_VIRTUAL_LINK3', 'LLEG_VIRTUAL_LINK2', 'LLEG_VIRTUAL_LINK1', 'RLEG_LINK0', 'RLEG_LINK1', 'RLEG_LINK2', 'RLEG_LINK3', 'RLEG_LINK4', 'RLEG_VIRTUAL_LINK3', 'RLEG_VIRTUAL_LINK2',
    'RLEG_VIRTUAL_LINK1', 'BODY', 'TORSO_VIRTUAL_LINK', 'HEAD_LINK0', 'LARM_LINK0', 'LARM_LINK1', 'LARM_LINK2', 'LARM_VIRTUAL_LINK2', 'LHAND_VIRTUAL_LINK', 'LARM_VIRTUAL_LINK1', 'RARM_LINK0', 'RARM_LINK1', 'RARM_LINK2', 'RARM_VIRTUAL_LINK2', 'RHAND_VIRTUAL_LINK', 'RARM_VIRTUAL_LINK1']
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

    # clone actor state in all of the environments
    for i in range(num_envs):

        state = gym.get_actor_rigid_body_states(
            envs[i], actor_handles[i], gymapi.STATE_NONE)
        state['pose']['p'].fill((root_pos.x, root_pos.y, root_pos.z))
        state['pose']['r'].fill(
            (root_quat.x, root_quat.y, root_quat.z, root_quat.w))
        gym.set_actor_rigid_body_states(
            envs[i], actor_handles[i], state, gymapi.STATE_ALL)

        gym.set_actor_dof_states(
            envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    frame += 1


print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
