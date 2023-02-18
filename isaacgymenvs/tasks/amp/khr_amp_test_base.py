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

from isaacgymenvs import ISAAC_GYM_ROOT_DIR
from isaacgymenvs.utils.torch_jit_utils import *
from ..base.vec_task import VecTask

# [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
# NUM_OBS = 1 + 6 + 3 + 3 + 16 + 16 + 12  # TODO: modify it for khr
NUM_OBS = 70
NUM_ACTIONS = 6  # head joint not included


# KEY_BODY_NAMES = ["RARM_LINK2", "LARM_LINK2", "RLEG_LINK4", "LLEG_LINK4"]
KEY_BODY_NAMES = ["LARM_LINK2", "RARM_LINK2"]


class KHRAMPTestBase(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = config

        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt

        # set control
        self._pd_control = self.cfg["env"]["control"]["pdControl"]
        self.Kp = self.cfg["env"]["control"]["Kp"]
        self.Kd = self.cfg["env"]["control"]["Kd"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.torque_limit = self.cfg["env"]["control"]["torqueLimit"]

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(
            self.sim)

        # TODO
        # sensors_per_env = 2
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(
        #     sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(
            dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 1]

        self._initial_dof_pos = torch.zeros_like(
            self._dof_pos, device=self.device, dtype=torch.float)
        # right_shoulder_x_handle = self.gym.find_actor_dof_handle(
        #     self.envs[0], self.khr_handles[0], "right_shoulder_x")
        # left_shoulder_x_handle = self.gym.find_actor_dof_handle(
        #     self.envs[0], self.khr_handles[0], "left_shoulder_x")  # TODO: match humanoid shoulder to khr
        # self._initial_dof_pos[:, right_shoulder_x_handle] = 0.5 * np.pi
        # self._initial_dof_pos[:, left_shoulder_x_handle] = -0.5 * np.pi

        self._initial_dof_vel = torch.zeros_like(
            self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13)[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(
            contact_force_tensor).view(self.num_envs, self.num_bodies, 3)

        # initilize some data used later on
        self.target_pos = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_vel = torch.zeros_like(
            self._dof_vel, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_target_pos = torch.zeros_like(self.target_pos,
                                                dtype=torch.float, device=self.device, requires_grad=False)
        self.torques = torch.zeros_like(
            self.dof_force_tensor, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions,
                                   dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_actions = torch.zeros_like(
            self.actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_dof_vel = torch.zeros_like(
            self._dof_vel, dtype=torch.float, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.projected_gravity_vec = torch.zeros_like(
            self.gravity_vec, dtype=torch.float, device=self.device, requires_grad=False)

        self._terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)

        if self.viewer != None:
            self._init_camera()

        return

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

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

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)

        return

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self.khr_handles[i]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '../../../assets')
        # asset_file = "urdf/khr/khr_set_limit_joint.urdf"
        asset_file = "urdf/khr/khr_set_limit_joint_w_small_foot_ver1.urdf"

        if "asset" in self.cfg["env"]:
            #asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get(
                "assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.angular_damping = 0.01  # TODO
        # asset_options.armature = 0.0001  # TODO
        # asset_options.angular_damping = 0.0
        # asset_options.linear_damping = 0.0  # TODO
        asset_options.max_angular_velocity = 100.0  # TODO
        asset_options.max_linear_velocity = 100.0  # TODO
        asset_options.collapse_fixed_joints = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # asset_options.thickness = 0.001  # TODO
        # asset_options.enable_gyroscopic_forces = True  # TODO
        # asset_options.replace_cylinder_with_capsule = True  # TODO
        khr_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)

        # actuator_props = self.gym.get_asset_actuator_properties(khr_asset)
        # motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(
            khr_asset, "RLEG_LINK4")
        left_foot_idx = self.gym.find_asset_rigid_body_index(
            khr_asset, "LLEG_LINK4")
        # sensor_pose = gymapi.Transform()

        # self.gym.create_asset_force_sensor(
        #     khr_asset, right_foot_idx, sensor_pose)
        # self.gym.create_asset_force_sensor(
        #     khr_asset, left_foot_idx, sensor_pose)

        # self.max_motor_effort = max(motor_efforts)
        # self.motor_efforts = to_torch(motor_efforts, device=self.device)

        # self.torso_index = 0  # TODO: modify for khr
        self.torso_index = self.gym.find_asset_rigid_body_index(
            khr_asset, "BODY")
        self.num_bodies = self.gym.get_asset_rigid_body_count(khr_asset)
        self.num_dof = self.gym.get_asset_dof_count(khr_asset)
        self.num_joints = self.gym.get_asset_joint_count(khr_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(
            *get_axis_params(0.89, self.up_axis_idx))  # TODO: modify for khr
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor(
            [start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.khr_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            contact_filter = 0

            handle = self.gym.create_actor(
                env_ptr, khr_asset, start_pose, "khr", i, contact_filter, 0)

            # self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))

            self.envs.append(env_ptr)
            self.khr_handles.append(handle)

            # if (self._pd_control):  # TODO
            # dof_prop = self.gym.get_asset_dof_properties(khr_asset)
            # dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            # self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

            dof_prop = self.gym.get_asset_dof_properties(khr_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_NONE
            self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(
            self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(
            self.dof_limits_upper, device=self.device)

        self.dof_pos_default = (
            self.dof_limits_lower + self.dof_limits_upper) / 2.0

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(
            env_ptr, handle)

        # TODO
        # if (self._pd_control):
        # self._build_pd_action_offset_scale()

        return

    # def _build_pd_action_offset_scale(self):  # TODO: modify
    #     num_joints = len(DOF_OFFSETS) - 1

    #     lim_low = self.dof_limits_lower.cpu().numpy()
    #     lim_high = self.dof_limits_upper.cpu().numpy()

    #     # for j in range(num_joints):
    #     #     dof_offset = DOF_OFFSETS[j]
    #     #     dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

    #     #     if (dof_size == 3):
    #     #         lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
    #     #         lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

    #     #     elif (dof_size == 1):
    #     for j in range(num_joints):
    #         curr_low = lim_low[j]
    #         curr_high = lim_high[j]
    #         curr_mid = 0.5 * (curr_high + curr_low)

    #         # extend the action range to be a bit beyond the joint limits so that the motors
    #         # don't lose their strength as they approach the joint limits
    #         curr_scale = 0.7 * (curr_high - curr_low)
    #         curr_low = curr_mid - curr_scale
    #         curr_high = curr_mid + curr_scale

    #         lim_low[j] = curr_low
    #         lim_high[j] = curr_high

    #     self._pd_action_offset = 0.5 * (lim_high + lim_low)
    #     self._pd_action_scale = 0.5 * (lim_high - lim_low)
    #     self._pd_action_offset = to_torch(
    #         self._pd_action_offset, device=self.device)
    #     self._pd_action_scale = to_torch(
    #         self._pd_action_scale, device=self.device)

    #     return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_khr_reward(
            self._root_states, self._dof_vel, self.prev_dof_vel, self.actions, self.prev_actions, self.torques, self.dt)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_khr_reset(self.reset_buf, self.progress_buf,
                                                                      self._contact_forces, self._contact_body_ids,
                                                                      self._rigid_body_pos, self.max_episode_length,
                                                                      self._enable_early_termination, self._termination_height)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_khr_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_khr_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            target_pos = self.target_pos
            actions = self.actions
            projected_gravity_vec = self.projected_gravity_vec
            # key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            target_pos = self.target_pos[env_ids]
            actions = self.actions[env_ids]
            projected_gravity_vec = self.projected_gravity_vec[env_ids]
            # key_body_pos = self._rigid_body_pos[env_ids][:,
            # self._key_body_ids, :]

        obs = compute_khr_observations(root_states, dof_pos, dof_vel,
                                       target_pos, actions, projected_gravity_vec, self.dof_limits_lower, self.dof_limits_upper)
        return obs

    def _reset_actors(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.prev_actions[env_ids] = 0.
        self.prev_dof_vel[env_ids] = 0.

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    # def pre_physics_step(self, actions):
    #     self.actions = actions.to(self.device).clone()

    #     if (self._pd_control):
    #         pd_tar = self._action_to_pd_targets(self.actions)
    #         pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
    #         self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
    #     else:
    #         forces = self.actions * \
    #             self.motor_efforts.unsqueeze(0) * self.power_scale
    #         force_tensor = gymtorch.unwrap_tensor(forces)
    #         self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    #     return

    def pre_physics_step(self, actions):

        self.actions[:] = actions.clone().to(self.device)

        # clip actions from output of mlp
        actions_clipped = torch.clamp(self.actions, min=-1.0, max=1.0)

        # scale actions and add prev_target_pos
        # target_pos = self.action_scale * actions_clipped + self.target_pos
        target_pos = unscale_dof_pos(
            actions_clipped, self.dof_limits_lower, self.dof_limits_upper)
        # target_pos = self.action_scale * target_pos + self.prev_target_pos

        # clip target_pos for dof limits
        self.target_pos = torch.clamp(
            target_pos, self.dof_limits_lower, self.dof_limits_upper)

        # if self.randomize_gain:
        #     Kp = self.Kp + to_
        #     Kd =
        # else:
        #     Kp = self.Kp
        #     Kd = self.Kd

        # calculate force tensor (torques) with torque limits
        self.torques[:] = set_pd_force_tensor_limit(
            self.Kp,
            self.Kd,
            self.target_pos,
            self._dof_pos,
            self.target_vel,
            self._dof_vel,
            self.torque_limit)

        # print('debug')
        # print("max force:", torch.max(self.dof_force_tensor),
        #       "min force:", torch.min(self.dof_force_tensor))

        # test for zero action
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(
        #     torch.zeros_like(force_tensor, dtype=torch.float, device=self.device, requires_grad=False)))

        # apply force tensor
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self.torques))  # apply actuator force

    def post_physics_step(self):
        self._refresh_sim_tensors()
        self.progress_buf += 1

        self.projected_gravity = quat_rotate_inverse(
            self._root_states[:, 3:7], self.gravity_vec)

        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # set prev buffer tensor
        self.prev_actions[:] = self.actions
        self.prev_dof_vel[:] = self._dof_vel

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render()
        return

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    # def _action_to_pd_targets(self, action):  # TODO: modify for khr
    #     pd_tar = self._pd_action_offset + self._pd_action_scale * action
    #     return pd_tar

    def _init_camera(self):  # TODO: modify for khr
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0],
                              self._cam_prev_char_pos[1] - 3.0,
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                  char_root_pos[1] + cam_delta[1],
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(
            self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################


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
    force_tensor = Kp * (target_pos - current_pos) + \
        Kd * (target_vel - current_vel)
    force_tensor = torch.clamp(
        force_tensor, min=-torque_limit, max=torque_limit)
    return force_tensor


@torch.jit.script
def scale_dof_pos(dof_pos, dof_limits_lower, dof_limits_upper):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    scaled_dof_pos = (2.0 * dof_pos - (dof_limits_lower +
                                       dof_limits_upper)) / (dof_limits_upper - dof_limits_lower)
    return scaled_dof_pos


@torch.jit.script
def unscale_dof_pos(scaled_dof_pos, dof_limits_lower, dof_limits_upper):
    unscaled_dof_pos = (dof_limits_upper + dof_limits_lower +
                        scaled_dof_pos * (dof_limits_upper - dof_limits_lower)) / 2.0
    return unscaled_dof_pos

# @torch.jit.script
# def dof_to_obs(pose):  # TODO: modify for khr
#     # type: (Tensor) -> Tensor
#     #dof_obs_size = 64
#     #dof_offsets = [0, 3, 6, 9, 12, 13, 16, 19, 20, 23, 24, 27, 30, 31, 34]
#     dof_obs_size = 52
#     dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
#     num_joints = len(dof_offsets) - 1

#     dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
#     dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
#     dof_obs_offset = 0

#     for j in range(num_joints):
#         dof_offset = dof_offsets[j]
#         dof_size = dof_offsets[j + 1] - dof_offsets[j]
#         joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

#         # assume this is a spherical joint
#         if (dof_size == 3):
#             joint_pose_q = exp_map_to_quat(joint_pose)
#             joint_dof_obs = quat_to_tan_norm(joint_pose_q)
#             dof_obs_size = 6
#         else:
#             joint_dof_obs = joint_pose
#             dof_obs_size = 1

#         dof_obs[:, dof_obs_offset:(
#             dof_obs_offset + dof_obs_size)] = joint_dof_obs
#         dof_obs_offset += dof_obs_size

#     return dof_obs


# @torch.jit.script
# TODO: modify for khr
# def compute_khr_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
#     # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
#     root_pos = root_states[:, 0:3]
#     root_rot = root_states[:, 3:7]
#     root_vel = root_states[:, 7:10]
#     root_ang_vel = root_states[:, 10:13]
#     root_h = root_pos[:, 2:3]
#     heading_rot = calc_heading_quat_inv(root_rot)
#     if (local_root_obs):
#         root_rot_obs = quat_mul(heading_rot, root_rot)
#     else:
#         root_rot_obs = root_rot
#     root_rot_obs = quat_to_tan_norm(root_rot_obs)
#     local_root_vel = my_quat_rotate(heading_rot, root_vel)
#     local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)
#     root_pos_expand = root_pos.unsqueeze(-2)
#     local_key_body_pos = key_body_pos - root_pos_expand
#     heading_rot_expand = heading_rot.unsqueeze(-2)
#     heading_rot_expand = heading_rot_expand.repeat(
#         (1, local_key_body_pos.shape[1], 1))
#     flat_end_pos = local_key_body_pos.view(
#         local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
#     flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
#                                                heading_rot_expand.shape[2])
#     local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
#     flat_local_key_pos = local_end_pos.view(
#         local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
#     # dof_obs = dof_to_obs(dof_pos)
#     # 1 + 6 + 3 + 3 + 16 + 16 + 12 = 57
#     obs = torch.cat((root_h, root_rot_obs, local_root_vel,
#                     local_root_ang_vel, dof_pos, dof_vel, flat_local_key_pos), dim=-1)
#     return obs
@torch.jit.script
def compute_khr_observations(root_states,
                             dof_pos,
                             dof_vel,
                             target_pos,
                             actions,
                             projected_gravity_vec,
                             dof_limits_lower,
                             dof_limits_upper
                             ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_rot = root_states[:, 3:7]
    root_ang_vel = root_states[:, 10:13]
    # roll, pitch, yaw = get_euler_xyz(base_quat)

    scaled_dof_pos = scale_dof_pos(
        dof_pos, dof_limits_lower=dof_limits_lower, dof_limits_upper=dof_limits_upper)
    scaled_target_pos = scale_dof_pos(
        target_pos, dof_limits_lower=dof_limits_lower, dof_limits_upper=dof_limits_upper)
    root_ang_vel = quat_rotate_inverse(
        root_rot, root_states[:, 10:13])

    obs = torch.cat((
        scaled_dof_pos,
        dof_vel,
        scaled_target_pos,
        projected_gravity_vec,
        actions,
        root_ang_vel,
    ), dim=-1)
    return obs


@torch.jit.script
def compute_khr_reward(root_states, dof_vel, prev_dof_vel, actions, prev_actions, torques, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor

    # set parameters to calculate rewards and penalties
    root_lin_vel = root_states[:, 7:10]
    lin_vel_x = root_lin_vel[:, 0] / 0.3

    # define rewards
    rew_lin_vel_x = torch.clamp(lin_vel_x, min=0.0, max=1.0)

    rewards = rew_lin_vel_x

    # define penalties
    pen_action_rate = torch.sum(torch.square(
        actions - prev_actions), dim=1) * 5e-3
    pen_dof_acc = torch.sum(torch.square(
        prev_dof_vel - dof_vel), dim=1) * 5e-4
    pen_torques = torch.sum(torch.square(
        torques), dim=1) * 0.0001
    pen_lin_vel_y = root_lin_vel[:, 1] * 0.001

    penalties = pen_action_rate + pen_dof_acc + pen_torques + pen_lin_vel_y

    # sum rewards and penalties
    total_rewards = rewards - penalties

    # clip
    total_rewards = torch.clip(total_rewards, min=0.)

    # reward = torch.ones_like(obs_buf[:, 0])
    return total_rewards


@torch.jit.script
def compute_khr_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                      max_episode_length, enable_early_termination, termination_height):  # TODO: modify for khr
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False

        # debug_height = body_height > 0.45
        # debug_height = torch.any(debug_height, dim=-1)
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)
        # has_fallen = torch.logical_or(debug_height, has_fallen)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(
            has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1,
                        torch.ones_like(reset_buf), terminated)

    return reset, terminated