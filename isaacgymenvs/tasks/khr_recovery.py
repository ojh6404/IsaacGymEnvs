# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict


class KHRRecovery(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        # 0.279252 = 16 * pi / 180
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["gravityVector"] = self.cfg["env"]["learn"]["gravityVectorRewardScale"]
        self.rew_scales["zHeight"] = self.cfg["env"]["learn"]["zHeightRewardScale"]
        self.rew_scales["posError"] = self.cfg["env"]["learn"]["posErrorRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["targetPosDiff"] = self.cfg["env"]["learn"]["targetPosDiffReward"]
        self.rew_scales["feetContact"] = self.cfg["env"]["learn"]["feetContactReward"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = state
        self.init_dropping_step = 50

        # base target state
        target_pos = self.cfg["env"]["baseTargetState"]["pos"]
        target_rot = self.cfg["env"]["baseTargetState"]["rot"]
        target_v_lin = self.cfg["env"]["baseTargetState"]["vLinear"]
        target_v_ang = self.cfg["env"]["baseTargetState"]["vAngular"]
        target_state = target_pos + target_rot + target_v_lin + target_v_ang
        self.base_target_state = target_state
        self.base_z_target = target_pos[2]

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # [joint_pos (10), prev_joint_target_pos(10)] * 3 + roll-pitch(2) + ang_vel(3) + phase(3)
        self.cfg["env"]["numObservations"] = 101
        self.cfg["env"]["numActions"] = 16       # only leg

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(
            self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        # self.dof_lower_limit = list(
        #     self.cfg["env"]["jointAnglesLowerLimit"].values())
        # self.dof_upper_limit = list(
        #     self.cfg["env"]["jointAnglesUpperLimit"].values())
        # TODO: set proper limit for each joint
        self.torque_limit = self.cfg["env"]["control"]["torqueLimit"]
        # TODO: set proper lower limit
        self.dof_lower_limit = to_torch(
            self.dof_lower_limit, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_upper_limit = to_torch(
            self.dof_upper_limit, dtype=torch.float, device=self.device, requires_grad=False)
        # TODO: set proper upper limit
        # if n, observation space will contain t, t-1, ..., t-n of action history
        self.prev_state_buffer_step = int(
            self.cfg["env"]["control"]["prevActionBuffer"])

        # for key in self.rew_scales.keys():
        # self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(
            torques).view(self.num_envs, self.num_dof)

        self.default_dof_pos = torch.zeros_like(
            self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            # deg to rad
            angle = self.named_default_joint_angles[name] / 180.0 * 3.14159
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(
            self.base_init_state, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions,
                                   dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_ref_target_pos = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_vel = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, dtype=torch.float, device=self.device, requires_grad=False)

        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.target_z_height = self.base_z_target * \
            torch.ones(self.num_envs, dtype=torch.float,
                       device=self.device, requires_grad=False)
        self.target_dof_pos = self.default_dof_pos

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                      self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '../../assets')
        # asset_file = "urdf/khr/khr.urdf"
        asset_file = "urdf/khr/khr_set_limit_joint_w_virtual_joint.urdf"

        # asset_path = os.path.join(asset_root, asset_file)
        # asset_root = os.path.dirname(asset_path)
        # asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]
        # asset_options.replace_cylinder_with_capsule = True
        # asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        # asset_options.density = 0.001
        asset_options.max_linear_velocity = 10.
        asset_options.max_angular_velocity = 10.
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.001
        asset_options.disable_gravity = False

        khr_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(khr_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(khr_asset)
        self.num_shapes = self.gym.get_asset_rigid_shape_count(khr_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # BODY, HEAD_LINK0, LARM_LINK0, ...
        body_names = self.gym.get_asset_rigid_body_names(khr_asset)
        self.dof_names = self.gym.get_asset_dof_names(khr_asset)
        feet_names = ["LLEG_LINK4", "RLEG_LINK4"]
        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = ["LLEG_LINK2", "RLEG_LINK2"]
        self.knee_indices = torch.zeros(
            len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(khr_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            # self.Kp
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"]
            # self.Kd
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"]

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.khr_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row)

            # self.gym.begin_aggregate(env_ptr, self.num_bodies, self.num_shapes, True)

            khr_handle = self.gym.create_actor(
                env_ptr, khr_asset, start_pose, "khr", i, 0, 0)  # self collision 1 to 0
            self.gym.set_actor_dof_properties(env_ptr, khr_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, khr_handle)

            # self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.khr_handles.append(khr_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.khr_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.khr_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.khr_handles[0], "BODY")

        print('base  names')
        print(self.dof_names)

        # Print DOF properties
        has_limits = dof_props['hasLimits']
        lower_limits = dof_props['lower']
        upper_limits = dof_props['upper']
        dof_types = [self.gym.get_asset_dof_type(
            khr_asset, i) for i in range(self.num_dof)]
        for i in range(self.num_dof):
            print("DOF %d" % i)
            print("  Name:     '%s'" % self.dof_names[i])
            print("  Type:     %s" %
                  self.gym.get_dof_type_string(dof_types[i]))
            print("  Limited?  %r" % has_limits[i])
            if has_limits[i]:
                print("    Lower   %f" % lower_limits[i])
                print("    Upper   %f" % upper_limits[i])

        self.dof_lower_limit = lower_limits
        self.dof_upper_limit = upper_limits

    def pre_physics_step(self, actions):
        # actions size : (num_envs, num_actions)
        self.actions = actions.clone().to(self.device)
        # clipped by -1.0 to 1.0
        self.actions = torch.clamp(self.actions, min=-1.0, max=1.0)
        self.actions = self.action_scale * self.actions + \
            self.prev_ref_target_pos    # action
        target_pos = dof_joint_limit_clip(
            self.actions, self.dof_lower_limit, self.dof_upper_limit)   # clipped by joiint limit
        force_tensor = set_pd_force_tensor_limit(
            self.Kp,
            self.Kd,
            target_pos,
            self.dof_pos,
            self.target_vel,
            self.dof_vel,
            self.torque_limit)

        # test for zero action
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torch.zeros_like(force_tensor, dtype=torch.float, device=self.device, requires_grad=False)))

        # apply force tensor clipped by torque limit
        # self.gym.set_dof_actuation_force_tensor(
        #     self.sim, gymtorch.unwrap_tensor(force_tensor))  # apply actuator force
        # self.prev_ref_target_pos = target_pos  # set previous target position

        if self.progress_buf[0] > self.init_dropping_step:
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(force_tensor))  # apply actuator force
            self.prev_ref_target_pos = target_pos  # set previous target position
        else:
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(
                torch.zeros_like(force_tensor, dtype=torch.float, device=self.device, requires_grad=False)))
            self.prev_ref_target_pos[:] = self.dof_pos[:]

        # normal PD control
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_khr_recovery_reward(
            # tensors
            self.obs_buf,
            self.root_states,
            self.dof_pos,
            self.gravity_vec,
            self.target_z_height,
            self.target_dof_pos,
            self.torques,
            self.contact_forces,
            self.feet_indices,
            self.progress_buf,
            # Dict
            self.rew_scales,
            # other
            self.base_index,
            self.num_actions,
            self.prev_state_buffer_step,
            self.init_dropping_step,
            self.max_episode_length,
            # self.base_target_state,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # print('obs_buf step -1')
        # print(self.obs_buf[:, 0: self.prev_state_buffer_step *
        #                    self.num_actions * 2])

        self.obs_buf[:] = compute_khr_recovery_observations(  # tensors
            self.obs_buf[:, 0: self.prev_state_buffer_step * \
                         self.num_actions * 2],
            self.dof_pos,
            self.prev_ref_target_pos,
            self.dof_lower_limit,
            self.dof_upper_limit,
            self.root_states,
            self.ang_vel_scale
        )

        # print('obs_buf step 0')
        # print(self.obs_buf[:, self.num_actions * 2: self.prev_state_buffer_step *
        #                    self.num_actions * 3])

    def reset_idx(self, env_ids):

        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

            # randomize initial quaternion states
            initial_root_euler_delta = torch_rand_float(
                -torch.pi / 2., torch.pi / 2., (self.num_envs, 2), device=self.device)
            initial_roll_delta, initial_pitch_delta = initial_root_euler_delta[
                :, 0], initial_root_euler_delta[:, 1]

            initial_roll, initial_pitch, initial_yaw = get_euler_xyz(
                self.initial_root_states[:, 3:7])
            initial_roll_rand, initial_pitch_rand = initial_roll + \
                initial_roll_delta, initial_pitch + initial_pitch_delta
            initial_root_quat_rand = quat_from_euler_xyz(
                initial_roll_rand, initial_pitch_rand, initial_yaw)
            initial_root_states = self.initial_root_states.detach()
            initial_root_states[:, 3:7] = initial_root_quat_rand

        else:
            initial_root_states = self.initial_root_states

        positions_offset = torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.01, 0.01,
                                      (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * \
            positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # reset observation buffer
        self.prev_ref_target_pos[env_ids] = self.dof_pos

        for i in range(self.prev_state_buffer_step + 1):
            self.obs_buf[env_ids, (2 * i) * self.num_actions: (2 + 2 * i) * self.num_actions] = \
                torch.cat((self.dof_pos, self.prev_ref_target_pos), dim=1)

        # self.obs_buf[env_ids, 2 * self.prev_state_buffer_step *
            # self.num_actions] = 0.

        # reset progress_buf & reset_buf
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

#####################################################################
###=========================jit functions=========================###
#####################################################################


@ torch.jit.script
def dof_joint_limit_clip(dof_state, dof_lower_limit, dof_upper_limit):
    clipped_dof_state = torch.max(
        torch.min(dof_state, dof_upper_limit), dof_lower_limit)
    return clipped_dof_state


@ torch.jit.script
def set_pd_force_tensor_limit(
    Kp,
    Kd,
    target_pos,
    current_pos,
    target_vel,
    current_vel,
    torque_limit
):
    # type: (float, float, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    force_tensor = Kp * (target_pos - current_pos) + \
        Kd * (target_vel - current_vel)
    force_tensor = torch.clamp(
        force_tensor, min=-torque_limit, max=torque_limit)
    return force_tensor


@ torch.jit.script
def compute_khr_recovery_reward(
        # tensor
        obs_buf,
        root_states,
        dof_pos,
        gravity_vec,
        target_z_height,
        target_dof_pos,
        torques,
        contact_forces,
        feet_indices,
        episode_lengths,
        # Dict
        rew_scales,
        # int
        base_index,
        num_actions,
        prev_state_buffer_step,
        init_dropping_step,
        max_episode_length,
        # base_target_state,
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, int, int, int) -> Tuple[Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_pos = root_states[:, :3]

    # upright reward
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    gravity_vec_error = torch.sum(torch.square(
        gravity_vec - projected_gravity), dim=1)
    rew_gravity_vector_error = torch.exp(-gravity_vec_error) * \
        rew_scales["gravityVector"]

    # height reward
    base_z_height = base_pos[:, 2]
    # rew_z_height = torch.zeros_like(base_z_height)
    # rew_z_height = torch.where(base_z_height > target_z_height, rew_z_height + 1., rew_z_height) * rew_scales["zHeight"]
    # rew_z_height_error = torch.pow(base_z_height - target_z_height, 2)
    # rew_z_height = torch.exp(-rew_z_height_error) * rew_scales["zHeight"]
    rew_z_height = base_z_height * rew_scales["zHeight"]

    # return init pos reward
    dof_pos_error = torch.sum(torch.square(dof_pos - target_dof_pos), dim=1)
    rew_dof_pos_error = torch.exp(-dof_pos_error) * rew_scales["posError"]

    # target joint pos difference
    # ahead_target_pos = obs_buf[:, 2 * num_actions * prev_state_buffer_step -
    #                            num_actions: 2 * num_actions * prev_state_buffer_step]
    # current_target_pos = obs_buf[:, 2 * num_actions * prev_state_buffer_step +
    #                              num_actions: 2 * num_actions * prev_state_buffer_step + 2 * num_actions]
    current_target_pos = obs_buf[:, num_actions: 2 * num_actions]
    ahead_target_pos = obs_buf[:, 2 * num_actions +
                               num_actions: 2 * num_actions + 2 * num_actions]
    target_pos_diff = torch.sum(torch.square(
        current_target_pos - ahead_target_pos), dim=1)
    rew_target_pos_diff = torch.exp(-target_pos_diff) * \
        rew_scales["targetPosDiff"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    # foot contact
    # reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    rew_feet_contact = rew_scales["feetContact"] * torch.all(torch.norm(
        contact_forces[:, feet_indices, :], dim=2) > 1., dim=1)

    total_reward = rew_torque + rew_gravity_vector_error + rew_z_height  # +
    # rew_dof_pos_error + # rew_target_pos_diff  # rew_feet_contact
    total_reward = torch.clip(total_reward, 0., None)

    if episode_lengths[0] < init_dropping_step:
        total_reward = torch.zeros_like(total_reward)

    # reset agents
    time_out = episode_lengths >= max_episode_length - \
        1  # no terminal reward for time-outs
    reset = time_out

    return total_reward.detach(), reset


@ torch.jit.script
def compute_khr_recovery_observations(prev_joint_obs_buf,
                                      dof_pos,
                                      prev_ref_target_pos,
                                      dof_lower_limit,
                                      dof_upper_limit,
                                      root_states,
                                      ang_vel_scale
                                      ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    base_quat = root_states[:, 3:7]
    roll, pitch, yaw = get_euler_xyz(base_quat)
    base_ang_vel = quat_rotate_inverse(
        base_quat, root_states[:, 10:13]) * ang_vel_scale
    joint_pos_scaled = (2.0 * dof_pos - (dof_lower_limit +
                                         dof_upper_limit)) / (dof_upper_limit - dof_lower_limit)
    prev_ref_target_pos_scaled = (2.0 * prev_ref_target_pos - (
        dof_lower_limit + dof_upper_limit)) / (dof_upper_limit - dof_lower_limit)

    # print(prev_joint_obs_buf)

    obs = torch.cat((
        joint_pos_scaled,
        prev_ref_target_pos_scaled,
        prev_joint_obs_buf,
        roll.unsqueeze(1),
        pitch.unsqueeze(1),
        base_ang_vel
    ), dim=-1)
    return obs
