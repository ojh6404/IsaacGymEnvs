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

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class KHR(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self.cfg["env"].get("angularVelocityScale", 0.1)
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        # self.death_cost = self.cfg["env"]["deathCost"]
        # self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
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

        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"] # deg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 101
        self.cfg["env"]["numActions"] = 16

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other setting
        self.dt = self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.torque_limit = 2.5 # TODO: set proper limit for each joint
        # self.dof_lower_limit = -1.5708 * torch.ones(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)    # TODO: set proper lower limit
        # self.dof_upper_limit = 1.5708 * torch.ones(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)    # TODO: set proper upper limit
        self.prev_size = 2 # if n, observation space will contain t, t-1, ..., t-n

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
            cam_target = gymapi.Vec3(0.3, 0.3, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        # self.initial_root_states[:, 7:13] = 0


        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        # self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        # zero_tensor = torch.tensor([0.0], device=self.device)
        # self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           # torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        # self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        # self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device, requires_grad=False)
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        # self.prev_potentials = self.potentials.clone()

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        # plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/khr/khr.urdf"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 100.0
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        khr_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Note - for this asset we are loading the actuator info from the MJCF
        # actuator_props = self.gym.get_asset_actuator_properties(khr_asset)
        # motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        # right_foot_idx = self.gym.find_asset_rigid_body_index(khr_asset, "right_foot")
        # left_foot_idx = self.gym.find_asset_rigid_body_index(khr_asset, "left_foot")
        # sensor_pose = gymapi.Transform()
        # self.gym.create_asset_force_sensor(khr_asset, right_foot_idx, sensor_pose)
        # self.gym.create_asset_force_sensor(khr_asset, left_foot_idx, sensor_pose)

        # self.max_motor_effort = max(motor_efforts)
        # self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.base_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(khr_asset)
        self.num_dof = self.gym.get_asset_dof_count(khr_asset)
        self.num_joints = self.gym.get_asset_joint_count(khr_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        # start_pose.p = gymapi.Vec3(*get_axis_params(1.34, self.up_axis_idx))
        # start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.khr_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            handle = self.gym.create_actor(env_ptr, khr_asset, start_pose, "khr", i, 0, 0)

            # self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)
            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.khr_handles.append(handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        # self.extremities = to_torch([5, 8], device=self.device, dtype=torch.long)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_khr_reward(
            # tensors
            self.root_states,
            # self.commands,
            self.torques,
            # self.contact_forces,
            self.knee_indices,
            self.progress_buf,
            # Dict
            # self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length,
        )

        # self.rew_buf[:], self.reset_buf = compute_khr_reward(
        #     self.obs_buf,
        #     self.reset_buf,
        #     self.progress_buf,
        #     self.actions,
        #     self.up_weight,
        #     self.heading_weight,
        #     self.potentials,
        #     self.prev_potentials,
        #     self.actions_cost_scale,
        #     self.energy_cost_scale,
        #     self.joints_at_limit_cost_scale,
        #     self.max_motor_effort,
        #     self.motor_efforts,
        #     self.termination_height,
        #     self.death_cost,
        #     self.max_episode_length
        # )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_khr_recovery_observations(  # tensors
                                                        self.obs_buf[:, : 2 * self.prev_size * self.num_actions],
                                                        self.dof_pos,
                                                        self.prev_ref_target_pos,
                                                        self.dof_lower_limit,
                                                        self.dof_upper_limit,
                                                        self.root_states,
                                                        self.ang_vel_scale
        )

        # self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_khr_observations(
        #     self.obs_buf, self.root_states, self.targets, self.potentials,
        #     self.inv_start_rot, self.dof_pos, self.dof_vel, self.torques,
        #     self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
        #     self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale, self.angular_velocity_scale,
        #     self.basis_vec0, self.basis_vec1)

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.01, 0.01, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        # velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        # self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        # self.dof_vel[env_ids] = velocities

        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.initial_root_states),
        #                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self.dof_state),
        #                                       gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.prev_ref_target_pos[env_ids] = self.dof_pos
        self.obs_buf[env_ids, 2 * self.prev_size * self.num_actions] = 0.

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)  # actions size : (num_envs, num_actions)
        self.actions = torch.clamp(self.actions, min=-1.0, max=1.0) # clipped by -1.0 to 1.0
        self.actions = self.action_scale * self.actions + self.prev_ref_target_pos    # action
        target_pos = dof_joint_limit_clip(self.actions, self.dof_lower_limit, self.dof_upper_limit)   # clipped by joiint limit
        force_tensor = set_pd_force_tensor_limit(
            self.Kp,
            self.Kd,
            target_pos,
            self.dof_pos,
            self.target_vel,
            self.dof_vel,
            self.torque_limit)
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torch.zeros_like(force_tensor, dtype=torch.float, device=self.device, requires_grad=False)))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(force_tensor)) # apply actuator force
        self.prev_ref_target_pos = target_pos  # set previous target position
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

        # self.actions = actions.to(self.device).clone()
        # forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        # force_tensor = gymtorch.unwrap_tensor(forces)
        # self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def dof_joint_limit_clip(dof_state, dof_lower_limit, dof_upper_limit):
    clipped_dof_state = torch.max(torch.min(dof_state, dof_upper_limit), dof_lower_limit)
    return clipped_dof_state

@torch.jit.script
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
    force_tensor = Kp * (target_pos - current_pos) + Kd * (target_vel - current_vel)
    force_tensor = torch.clamp(force_tensor, min=-torque_limit, max=torque_limit)
    return force_tensor


@torch.jit.script
def compute_khr_reward(
    # tensor
    root_states,
    torques,
    knee_indices,
    episode_lengths,
    base_index,
    max_episode_length
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, int, int) -> Tuple[Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    # lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    # ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    # rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    # rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * -0.000025

    total_reward = rew_torque
    total_reward = torch.clip(total_reward, 0., None)
    # reset agents
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = time_out

    return total_reward.detach(), reset

@torch.jit.script
def compute_khr_observations(prev_joint_obs_buf,
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
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    joint_pos_scaled = (2.0 * dof_pos - (dof_lower_limit + dof_upper_limit)) / (dof_upper_limit - dof_lower_limit)
    prev_ref_target_pos_scaled = (2.0 * prev_ref_target_pos - (dof_lower_limit + dof_upper_limit)) / (dof_upper_limit - dof_lower_limit)

    obs = torch.cat((prev_joint_obs_buf,
                     joint_pos_scaled,
                     prev_ref_target_pos_scaled,
                     roll.unsqueeze(1),
                     pitch.unsqueeze(1),
                     base_ang_vel
                     ), dim=-1)

    return obs


# @torch.jit.script
# def compute_khr_reward(
#     obs_buf,
#     reset_buf,
#     progress_buf,
#     actions,
#     up_weight,
#     heading_weight,
#     potentials,
#     prev_potentials,
#     actions_cost_scale,
#     energy_cost_scale,
#     joints_at_limit_cost_scale,
#     max_motor_effort,
#     motor_efforts,
#     termination_height,
#     death_cost,
#     max_episode_length
# ):
#     # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, Tensor, float, float, float) -> Tuple[Tensor, Tensor]

#     # reward from the direction headed
#     heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
#     heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

#     # reward for being upright
#     up_reward = torch.zeros_like(heading_reward)
#     up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

#     actions_cost = torch.sum(actions ** 2, dim=-1)

#     # energy cost reward
#     motor_effort_ratio = motor_efforts / max_motor_effort
#     scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 12:33]) - 0.98) / 0.02
#     dof_at_limit_cost = torch.sum((torch.abs(obs_buf[:, 12:33]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1)

#     electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 33:54]) * motor_effort_ratio.unsqueeze(0), dim=-1)

#     # reward for duration of being alive
#     alive_reward = torch.ones_like(potentials) * 2.0
#     progress_reward = potentials - prev_potentials

#     total_reward = progress_reward + alive_reward + up_reward + heading_reward - \
#         actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost

#     # adjust reward for fallen agents
#     total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

#     # reset agents
#     reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
#     reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

#     return total_reward, reset


# @torch.jit.script
# def compute_khr_observations(obs_buf, root_states, targets, potentials, inv_start_rot, dof_pos, dof_vel,
#                                   dof_force, dof_limits_lower, dof_limits_upper, dof_vel_scale,
#                                   sensor_force_torques, actions, dt, contact_force_scale, angular_velocity_scale,
#                                   basis_vec0, basis_vec1):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

#     torso_position = root_states[:, 0:3]
#     torso_rotation = root_states[:, 3:7]
#     velocity = root_states[:, 7:10]
#     ang_velocity = root_states[:, 10:13]

#     to_target = targets - torso_position
#     to_target[:, 2] = 0

#     prev_potentials_new = potentials.clone()
#     potentials = -torch.norm(to_target, p=2, dim=-1) / dt

#     torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
#         torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

#     vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
#         torso_quat, velocity, ang_velocity, targets, torso_position)

#     roll = normalize_angle(roll).unsqueeze(-1)
#     yaw = normalize_angle(yaw).unsqueeze(-1)
#     angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)
#     dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

#     # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs (21), num_dofs (21), 6, num_acts (21)
#     obs = torch.cat((torso_position[:, 2].view(-1, 1), vel_loc, angvel_loc * angular_velocity_scale,
#                      yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
#                      dof_pos_scaled, dof_vel * dof_vel_scale, dof_force * contact_force_scale,
#                      sensor_force_torques.view(-1, 12) * contact_force_scale, actions), dim=-1)

#     return obs, potentials, prev_potentials_new, up_vec, heading_vec
