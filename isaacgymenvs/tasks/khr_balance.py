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

from typing import Tuple, Dict


class KHRBalance(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["upright"] = self.cfg["env"]["learn"]["uprightRewardScale"]
        self.rew_scales["heading"] = self.cfg["env"]["learn"]["headingRewardScale"]
        self.rew_scales["dofPosError"] = self.cfg["env"]["learn"]["dofPosErrorRewardScale"]

        # penalty
        self.rew_scales["torques"] = self.cfg["env"]["learn"]["torques"]
        self.rew_scales["linearVelocity"] = self.cfg["env"]["learn"]["linearVelocity"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # debug
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

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
        self.random_initialize = self.cfg["env"]["randomInitialize"]

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

        # [joint_pos (16), prev_target_pos (16)] + roll-pitch(2) + ang_vel(3)
        self.cfg["env"]["numObservations"] = 37
        self.cfg["env"]["numActions"] = 16       # head joint not included

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # set time and episode length
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(
            self.max_episode_length_s / self.dt + 0.5)

        # set push robot
        self.push_robot = self.cfg["env"]["learn"]["pushRobots"]
        self.push_interval = int(
            self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)

        # scaling rewards
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        # set gain and torque limits for control
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.torque_limit = self.cfg["env"]["control"]["torqueLimit"]

        # set dof limits
        self.prev_state_buffer_step = int(
            self.cfg["env"]["control"]["prevActionBuffer"])

        # set viewer
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
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # refresh tensor
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

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
        self.rb_states = gymtorch.wrap_tensor(
            rb_state_tensor).view(self.num_envs, -1, 13)
        self.rb_pos = self.rb_states[..., 0:3]
        self.rb_quats = self.rb_states[..., 3:7]

        # set default dof pos
        self.default_dof_pos = torch.zeros_like(
            self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(len(self.dof_names)):
            name = self.dof_names[i]
            # deg to rad
            angle = self.named_default_joint_angles[name] / 180.0 * torch.pi
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(
            self.base_init_state, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions,
                                   dtype=torch.float, device=self.device, requires_grad=False)
        self.force_tensor = torch.zeros_like(
            self.torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_dof_vel = torch.zeros_like(
            self.dof_vel, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_vel = torch.zeros_like(
            self.dof_vel, dtype=torch.float, device=self.device, requires_grad=False)
        # self.dof_acc = torch.zeros_like(
        #     self.dof_vel, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_pos = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_actions = torch.zeros_like(self.actions,
                                             dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_target_pos = torch.zeros_like(self.target_pos,
                                                dtype=torch.float, device=self.device, requires_grad=False)

        self.target_z_height = self.base_z_target * \
            torch.ones(self.num_envs, dtype=torch.float,
                       device=self.device, requires_grad=False)
        self.target_dof_pos = self.default_dof_pos

        # set vectors for env
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.up_vec = to_torch(get_axis_params(
            1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.x_axis_vec = to_torch(
            [1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.y_axis_vec = to_torch(
            [0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.z_axis_vec = self.up_vec

        # reset env
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

        # asset_path = os.path.join(asset_root, asset_file)
        # asset_root = os.path.dirname(asset_path)
        # asset_file = os.path.basename(asset_path)

        asset_root = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/khr/khr_set_limit_joint.urdf"

        # dof_names : ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LLEG_JOINT0', 'LLEG_JOINT1', 'LLEG_JOINT2', 'LLEG_JOINT3', 'LLEG_JOINT4', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RLEG_JOINT0', 'RLEG_JOINT1', 'RLEG_JOINT2', 'RLEG_JOINT3', 'RLEG_JOINT4']
        # body_names : ['BODY', 'LARM_LINK0', 'LARM_LINK1', 'LARM_LINK2', 'LLEG_LINK0', 'LLEG_LINK1', 'LLEG_LINK2', 'LLEG_LINK3', 'LLEG_LINK4', 'RARM_LINK0', 'RARM_LINK1', 'RARM_LINK2', 'RLEG_LINK0', 'RLEG_LINK1', 'RLEG_LINK2', 'RLEG_LINK3', 'RLEG_LINK4']

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.max_linear_velocity = self.cfg["env"]["assetOptions"]["maxLinearVelocity"]
        asset_options.max_angular_velocity = self.cfg["env"]["assetOptions"]["maxAngularVelocity"]
        asset_options.linear_damping = self.cfg["env"]["assetOptions"]["linearDamping"]
        asset_options.angular_damping = self.cfg["env"]["assetOptions"]["angularDamping"]
        asset_options.armature = self.cfg["env"]["assetOptions"]["armature"]
        asset_options.thickness = self.cfg["env"]["assetOptions"]["thickness"]
        asset_options.disable_gravity = self.cfg["env"]["assetOptions"]["disableGravity"]
        # asset_options.replace_cylinder_with_capsule = True
        # asset_options.flip_visual_attachments = True

        khr_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(khr_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(khr_asset)
        self.num_shapes = self.gym.get_asset_rigid_shape_count(khr_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # link names, BODY, HEAD_LINK0, LARM_LINK0, ...
        self.body_names = self.gym.get_asset_rigid_body_names(khr_asset)
        self.dof_names = self.gym.get_asset_dof_names(khr_asset)
        self.feet_names = ["LLEG_LINK4", "RLEG_LINK4"]
        self.other_names = [
            body_name for body_name in self.body_names if body_name not in self.feet_names]
        self.feet_indices = torch.zeros(
            len(self.feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.body_indices = torch.zeros(
            len(self.body_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.other_indices = torch.zeros(
            len(self.other_names), dtype=torch.long, device=self.device, requires_grad=False)

        dof_props = self.gym.get_asset_dof_properties(khr_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_NONE
            # self.Kp
            # dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"]
            # self.Kd
            # dof_props['damping'][i] = self.cfg["env"]["control"]["damping"]

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
                env_ptr, khr_asset, start_pose, "khr", i, 0, 0)  # self-collision 1 to 0
            self.gym.set_actor_dof_properties(env_ptr, khr_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, khr_handle)

            self.envs.append(env_ptr)
            self.khr_handles.append(khr_handle)

        for i in range(len(self.feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.khr_handles[0], self.feet_names[i])
        for i in range(len(self.body_names)):
            self.body_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.khr_handles[0], self.body_names[i])
        for i in range(len(self.other_names)):
            self.other_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.khr_handles[0], self.other_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.khr_handles[0], "BODY")

        # Print DOF properties
        has_limits = dof_props['hasLimits']
        dof_lower_limits = dof_props['lower']
        dof_upper_limits = dof_props['upper']
        dof_types = [self.gym.get_asset_dof_type(
            khr_asset, i) for i in range(self.num_dof)]
        for i in range(self.num_dof):
            print("DOF %d" % i)
            print("  Name:     '%s'" % self.dof_names[i])
            print("  Type:     %s" %
                  self.gym.get_dof_type_string(dof_types[i]))
            print("  Limited?  %r" % has_limits[i])
            if has_limits[i]:
                print("    Lower   %f" % dof_lower_limits[i])
                print("    Upper   %f" % dof_upper_limits[i])

        # set dof limits
        self.dof_lower_limits = to_torch(
            dof_lower_limits, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_upper_limits = to_torch(
            dof_upper_limits, dtype=torch.float, device=self.device, requires_grad=False)

        # set init dof limits for avoidng self collision when initializing
        self.init_dof_lower_limits = self.dof_lower_limits[:]
        self.init_dof_upper_limits = self.dof_upper_limits[:]
        self.init_dof_lower_limits[3] = 0.
        self.init_dof_upper_limits[11] = 0.
        self.init_dof_lower_limits[1] = 0.
        self.init_dof_upper_limits[9] = 0.

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # clip actions from output of mlp
        target_pos = torch.clamp(self.actions, min=-1.0, max=1.0)

        # scale actions and add prev_target_pos
        # target_pos = self.action_scale * target_pos + self.default_dof_pos
        target_pos = self.action_scale * target_pos + self.prev_target_pos

        # target_pos = self.default_dof_pos

        # clip target_pos for dof limits
        target_pos = torch.clamp(
            target_pos, self.dof_lower_limits, self.dof_upper_limits)

        # calculate force tensor (torques) with torque limits
        self.force_tensor[:] = set_pd_force_tensor_limit(
            self.Kp,
            self.Kd,
            target_pos,
            self.dof_pos,
            self.target_vel,
            self.dof_vel,
            self.torque_limit)

        # print('debug')
        # print("max force:", torch.max(self.force_tensor),
        #       "min force:", torch.min(self.force_tensor))

        # test for zero action
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torch.zeros_like(force_tensor, dtype=torch.float, device=self.device, requires_grad=False)))

        # apply force tensor
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self.force_tensor))  # apply actuator force

        # set previous target position
        self.prev_target_pos[:] = target_pos

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # push robots
        self.common_step_counter += 1
        if self.push_robot and (self.common_step_counter % self.push_interval == 0):
            self.push_robots()

        self.compute_observations()
        self.compute_reward(self.actions)

        # set prev history buffer
        self.prev_actions[:] = self.actions

        # debug
        if self.viewer and self.debug_viz:
            self.axis_visualiztion(self.feet_indices)

    def push_robots(self):
        self.root_states[:, 7:9] = torch_rand_float(
            -1.0, 1.0, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_khr_balance_reward(
            # tensors
            self.obs_buf,
            self.root_states,
            self.rb_states,
            self.dof_pos,
            self.dof_vel,
            self.dof_acc,
            self.target_z_height,
            self.up_vec,
            self.target_dof_pos,
            self.force_tensor,
            self.contact_forces,
            self.feet_indices,
            self.other_indices,
            self.progress_buf,
            self.actions,
            self.prev_actions,
            # Dict
            self.rew_scales,
            # other
            self.base_index,
            self.num_actions,
            self.prev_state_buffer_step,
            self.max_episode_length,
            # self.base_target_state,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # compute acceleration
        self.dof_acc = compute_dof_acceleration(
            self.dof_vel, self.prev_dof_vel, self.dt)

        self.obs_buf[:] = compute_khr_balance_observations(  # tensors
            scale_joint_pos(self.dof_pos, self.dof_lower_limits,
                            self.dof_upper_limits),
            scale_joint_pos(self.prev_target_pos,
                            self.dof_lower_limits, self.dof_upper_limits),
            self.root_states,
            self.ang_vel_scale
        )

    def reset_idx(self, env_ids):

        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # randomize initial quaternion states
        if self.random_initialize:
            initial_root_quat_delta = torch_rand_float(
                -0.1, 0.1, (self.num_envs, 4), device=self.device)
            initial_root_quat_delta[:, 0] = torch_rand_float(
                -0.05, 0.05, (self.num_envs, 1), device=self.device).squeeze()  # for roll
            initial_root_quat_delta[:, 2:4] = 0.  # for yaw
            initial_root_quat_rand = self.initial_root_states[:,
                                                              3:7] + initial_root_quat_delta
            initial_root_states = self.initial_root_states.detach()
            initial_root_states[:, 3:7] = quat_unit(initial_root_quat_rand)
        else:
            initial_root_states = self.initial_root_states

        # randomize dof_pos
        positions_offset = torch_rand_float(
            - torch.pi / 6., torch.pi / 6., (len(env_ids), self.num_dof), device=self.device)
        # arm randomization
        positions_offset[:, 0:3] = torch_rand_float(
            - torch.pi, torch.pi, (len(env_ids), 3), device=self.device)
        positions_offset[:, 8:11] = torch_rand_float(
            - torch.pi, torch.pi, (len(env_ids), 3), device=self.device)
        # randomize dof_vel
        velocities = torch_rand_float(-1.0, 1.0,
                                      (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = torch.clamp(
            positions_offset, min=self.init_dof_lower_limits, max=self.init_dof_upper_limits)
        self.dof_vel[env_ids] = velocities

        # set initial dof_state
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # reset prev target pos
        self.prev_target_pos[env_ids] = self.dof_pos[env_ids]

        # reset obs buffer for dof pos and prev target dof pos
        # for i in range(self.prev_state_buffer_step + 1):
        #     self.obs_buf[env_ids, (2 * i) * self.num_actions: (2 + 2 * i) * self.num_actions] = \
        #         torch.cat((scale_joint_pos(self.dof_pos[env_ids], self.dof_lower_limits, self.dof_upper_limits),
        #                   scale_joint_pos(self.prev_target_pos[env_ids], self.dof_lower_limits, self.dof_upper_limits)), dim=1)

        # # reset obs buffer for roll, pitch
        # self.obs_buf[env_ids, self.num_actions * 2 * (self.prev_state_buffer_step + 1): self.num_actions * 2 * (
        #     self.prev_state_buffer_step + 1) + 2] = torch.hstack([initial_roll_rand[env_ids].unsqueeze(1), initial_pitch_rand[env_ids].unsqueeze(1)])

        # # reset obs buffer for base_ang_vel
        # self.obs_buf[env_ids, self.num_actions * 2 * (self.prev_state_buffer_step + 1) + 2: self.num_actions * 2 * (
        #     self.prev_state_buffer_step + 1) + 5] = initial_root_states[env_ids, 10:13] * self.ang_vel_scale

        # reset progress_buf & reset_buf
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def axis_visualiztion(self, rb_indices):
        # compute start and end positions for visualizing thrust lines
        env_offsets = torch.zeros(
            (self.num_envs, len(rb_indices), 3), device=self.device)

        for i in range(self.num_envs):
            env_origin = self.gym.get_env_origin(self.envs[i])
            env_offsets[i, ..., 0] = env_origin.x
            env_offsets[i, ..., 1] = env_origin.y
            env_offsets[i, ..., 2] = env_origin.z

        rb_pos_indexed = self.rb_pos[:, rb_indices, :]
        rb_quat_indexed = self.rb_quats[:, rb_indices, :]

        glob_pos = rb_pos_indexed + env_offsets

        axis_vec_scale = 0.3
        x_vec = quat_axis(rb_quat_indexed.view(self.num_envs * len(rb_indices), 4),
                          0).view(self.num_envs, len(rb_indices), 3) * axis_vec_scale
        y_vec = quat_axis(rb_quat_indexed.view(self.num_envs * len(rb_indices), 4),
                          1).view(self.num_envs, len(rb_indices), 3) * axis_vec_scale
        z_vec = quat_axis(rb_quat_indexed.view(self.num_envs * len(rb_indices), 4),
                          2).view(self.num_envs, len(rb_indices), 3) * axis_vec_scale

        x_end = glob_pos + x_vec
        y_end = glob_pos + y_vec
        z_end = glob_pos + z_vec

        # submit debug line geometry
        x_vert = torch.stack([glob_pos, x_end], dim=2)
        y_vert = torch.stack([glob_pos, y_end], dim=2)
        z_vert = torch.stack([glob_pos, z_end], dim=2)
        xyz_verts = torch.stack([x_vert, y_vert, z_vert], dim=2).cpu().numpy()

        colors = np.zeros((
            self.num_envs, len(rb_indices), 3, 3), dtype=np.float32)
        colors[:, :, 0] = [1., 0., 0.]
        colors[:, :, 1] = [0., 1., 0.]
        colors[:, :, 2] = [0., 0., 1.]

        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(self.viewer, None, self.num_envs *
                           len(rb_indices) * 3, xyz_verts, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_khr_balance_reward(
        # tensor
        obs_buf,
        root_states,
        rb_states,
        dof_pos,
        dof_vel,
        dof_acc,
        target_z_height,
        up_vec,
        target_dof_pos,
        torques,
        contact_forces,
        feet_indices,
        other_indices,
        episode_lengths,
        actions,
        prev_actions,
        # Dict
        rew_scales,
        # int
        base_index,
        num_actions,
        prev_state_buffer_step,
        max_episode_length,
    # (reward, reset, feet_in air, feet_air_time, episode sums)
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, int, int) -> Tuple[Tensor, Tensor]

    base_pos = root_states[:, :3]
    base_quat = root_states[:, 3:7]

    # world frame, if quat_rotate_inverse(base_quat, base_lin_vel), local base
    base_lin_vel = root_states[:, 7:10]
    base_ang_vel = root_states[:, 10:13]  # world frame

    num_envs = base_pos.shape[0]
    up_vec = get_basis_vector(base_quat, up_vec).view(num_envs, 3)
    up_vec_norm_xy = torch.norm(up_vec[:, :2], dim=1)
    up_proj = up_vec[:, 2]

    phi = torch.atan2(up_vec_norm_xy, up_proj)
    heading = calc_heading(base_quat)

    dof_pos_error = torch.norm(target_dof_pos - dof_pos, dim=1)

    # from https://arxiv.org/pdf/1809.02074.pdf
    alpha_phi = 5.
    alpha_heading = 5.
    alpha_root_x_pos = 2000.
    alpha_root_y_pos = 2000.
    alpha_root_z_pos = 2000.
    alpha_root_x_vel = 5.
    alpha_root_y_vel = 5.
    alpha_root_z_vel = 5.
    alpha_dof_pos_error = 1.
    alpha_dof_vel = 1.

    # upright root
    # rew_upright_root = torch.exp(-alpha_phi * phi ** 2) * rew_scales["upright"]
    # rew_heading_root = torch.exp(-alpha_heading *
    # heading ** 2) * rew_scales["heading"]
    # rew_root_x_pos = torch.exp(-alpha_root_x_pos *
    #                            (base_pos[:, 0]) ** 2) * rew_scales["xPos"]
    # rew_root_y_pos = torch.exp(-alpha_root_y_pos *
    #                            (base_pos[:, 1]) ** 2) * rew_scales["yPos"]
    # rew_root_z_pos = torch.exp(-alpha_root_z_pos *
    # (target_z_height - base_pos[:, 2]) ** 2) * rew_scales["zPos"]
    # rew_root_x_vel = torch.exp(-alpha_root_x_vel *
    #                            (base_lin_vel[:, 0]) ** 2) * rew_scales["xVel"]
    # rew_root_y_vel = torch.exp(-alpha_root_y_vel *
    #                            (base_lin_vel[:, 1]) ** 2) * rew_scales["yVel"]
    # rew_root_z_vel = torch.exp(-alpha_root_z_vel *
    #                            (base_lin_vel[:, 2]) ** 2) * rew_scales["zVel"]
    # rew_dof_pos_error = torch.exp(-alpha_dof_pos_error *
    #                               dof_pos_error) * rew_scales["dofPosError"]
    # rew_dof_vel = torch.exp(-alpha_dof_vel * dof_vel) * rew_scales["dofVel"]

    # pen_torques = torch.sum(torch.square(torques), dim=1) * 0.001

    # # calculate total reward
    # total_reward = \
    #     rew_upright_root + rew_heading_root + rew_root_x_pos + rew_root_y_pos + \
    #     rew_root_z_pos + rew_root_x_vel + rew_root_y_vel + \
    #     rew_root_z_vel + rew_dof_pos_error + rew_dof_vel - pen_torques
    # total_reward = torch.clip(total_reward, 0., None)

    rew_upright_root = (torch.pi / 2. - phi) / \
        (torch.pi / 2.) * rew_scales["upright"]
    rew_heading_root = (torch.pi - torch.abs(heading)) / \
        (torch.pi) * rew_scales["heading"]
    rew_dof_pos_error = torch.exp(- 1.0 * dof_pos_error) * \
        rew_scales["dofPosError"]

    pen_root_lin_vel = torch.sum(torch.square(
        base_lin_vel), dim=1) * rew_scales["linearVelocity"]
    # pen_root_ang_vel = torch.sum(torch.square(base_ang_vel), dim=1) * 0.5
    # pen_dof_vel = torch.sum(torch.square(dof_vel), dim=1) * 0.001
    pen_torques = torch.sum(torch.square(
        torques), dim=1) * rew_scales["torques"]
    # pen_actions = torch.sum(torch.where(torch.abs(
    #     actions) > 1.0, 1. * torch.ones_like(actions), torch.zeros_like(actions)), dim=1) * 0.01

    # calculate total reward
    # total_reward = rew_dof_pos_error - pen_actions
    penalty = pen_root_lin_vel + pen_torques
    total_reward = rew_upright_root + rew_heading_root + rew_dof_pos_error - penalty

    total_reward = torch.clip(total_reward, 0., None)
    # reset when time out
    time_out = episode_lengths >= max_episode_length - \
        1
    reset = time_out
    base_limit = base_pos[:, 2] < 0.20
    reset = reset | base_limit

    # print("debug")
    # print("rew_upright_root", rew_upright_root[0])
    # print("rew_heading_root", rew_heading_root[0])
    # print("rew_root_x_pos", rew_root_x_pos[0])
    # print("rew_root_y_pos", rew_root_y_pos[0])
    # print("rew_root_z_pos", rew_root_z_pos[0])
    # print("rew_root_x_vel", rew_root_x_vel[0])
    # print("rew_root_y_vel", rew_root_y_vel[0])
    # print("rew_root_z_vel", rew_root_z_vel[0])
    # print("rew_dof_pos_error", rew_dof_pos_error[0])
    # print("rew_dof_vel", rew_dof_vel[0])
    # print("pen_torques", pen_torques[0])
    # print("max actions", torch.max(actions[0]))
    # print("min actions", torch.min(actions[0]))

    # print("debug")
    # print("rew_upright_root", rew_upright_root[0])
    # print("rew_heading_root", rew_heading_root[0])
    # print("rew_dof_pos_error", rew_dof_pos_error[0])
    # print("dof_pos", dof_pos[0])
    # # print("target_dof_pos", target_dof_pos[0])
    # # print("target_dof_pos", target_dof_pos)
    # # print("pen_dof_vel", pen_dof_vel[0])
    # # print("pen_torques", pen_torques[0])
    # print("pen_actions", pen_actions[0])
    # print("max action", torch.max(actions[0]))
    # print("min action", torch.min(actions[0]))

    return total_reward.detach(), reset


@ torch.jit.script
def compute_khr_balance_observations(dof_pos_scaled,
                                     prev_target_pos_scaled,
                                     root_states,
                                     ang_vel_scale
                                     ):

    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    base_quat = root_states[:, 3:7]
    roll, pitch, yaw = get_euler_xyz(base_quat)
    base_ang_vel = quat_rotate_inverse(
        base_quat, root_states[:, 10:13]) * ang_vel_scale

    obs = torch.cat((
        dof_pos_scaled,
        prev_target_pos_scaled,
        roll.unsqueeze(1),
        pitch.unsqueeze(1),
        base_ang_vel
    ), dim=-1)
    return obs


@torch.jit.script
def compute_dof_acceleration(dof_vel, prev_dof_vel, dt):
    # type: (Tensor, Tensor, float) -> Tensor
    dof_acc = (dof_vel - prev_dof_vel) / dt
    return dof_acc


@torch.jit.script
def scale_joint_pos(dof_pos, dof_lower_limit, dof_upper_limit):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    joint_pos_scaled = (2.0 * dof_pos - (dof_lower_limit +
                                         dof_upper_limit)) / (dof_upper_limit - dof_lower_limit)
    return joint_pos_scaled


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
