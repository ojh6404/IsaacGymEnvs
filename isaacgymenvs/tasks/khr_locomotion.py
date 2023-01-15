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


class KHRLocomotion(VecTask):

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
        self.rew_scales["feetContact"] = self.cfg["env"]["learn"]["feetContactRewardScale"]
        self.rew_scales["feetUpright"] = self.cfg["env"]["learn"]["feetUprightRewardScale"]

        # penalty scales
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torquePenaltyScale"]
        self.rew_scales["power"] = self.cfg["env"]["learn"]["powerPenaltyScale"]
        self.rew_scales["dofVelocity"] = self.cfg["env"]["learn"]["dofVelocityPenaltyScale"]
        self.rew_scales["dofAcceleration"] = self.cfg["env"]["learn"]["dofAccelerationPenaltyScale"]
        self.rew_scales["actionsDiff"] = self.cfg["env"]["learn"]["actionsDiffPenaltyScale"]
        self.rew_scales["bodyContact"] = self.cfg["env"]["learn"]["bodyContactPenaltyScale"]
        self.rew_scales["contactForces"] = self.cfg["env"]["learn"]["contactForcesPenaltyScale"]
        self.rew_scales["baseLinearVelocity"] = self.cfg["env"]["learn"]["baseLinearVelocityPenaltyScale"]
        self.rew_scales["baseAngularVelocity"] = self.cfg["env"]["learn"]["baseAngularVelocityPenaltyScale"]
        self.rew_scales["targetPosDiff"] = self.cfg["env"]["learn"]["targetPosDiffPenaltyScale"]

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
        self.init_dropping_step = self.cfg["env"]["baseInitState"]["initDroppingStep"]
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
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

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
        self.prev_actions = torch.zeros_like(self.actions,
                                             dtype=torch.float, device=self.device, requires_grad=False)
        self.target_pos = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_target_pos = torch.zeros_like(self.target_pos,
                                                dtype=torch.float, device=self.device, requires_grad=False)
        self.force_tensor = torch.zeros_like(
            self.torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_vel = torch.zeros_like(
            self.dof_vel, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_dof_vel = torch.zeros_like(
            self.dof_vel, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_acc = torch.zeros_like(
            self.dof_vel, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, dtype=torch.float, device=self.device, requires_grad=False)

        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.target_z_height = self.base_z_target * \
            torch.ones(self.num_envs, dtype=torch.float,
                       device=self.device, requires_grad=False)
        self.target_dof_pos = self.default_dof_pos

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self.up_vec = to_torch(get_axis_params(
            1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.x_axis_vec = to_torch(
            [1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.y_axis_vec = to_torch(
            [0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.z_axis_vec = self.up_vec

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
        asset_options.max_linear_velocity = self.cfg["env"]["assetOptions"]["maxLinearVelocity"]
        asset_options.max_angular_velocity = self.cfg["env"]["assetOptions"]["maxAngularVelocity"]
        asset_options.linear_damping = self.cfg["env"]["assetOptions"]["linearDamping"]
        asset_options.angular_damping = self.cfg["env"]["assetOptions"]["angularDamping"]
        asset_options.armature = self.cfg["env"]["assetOptions"]["armature"]
        # 0.001
        asset_options.thickness = self.cfg["env"]["assetOptions"]["thickness"]
        asset_options.disable_gravity = self.cfg["env"]["assetOptions"]["disableGravity"]

        khr_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(khr_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(khr_asset)
        self.num_shapes = self.gym.get_asset_rigid_shape_count(khr_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # BODY, HEAD_LINK0, LARM_LINK0, ...
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
        # knee_names = ["LLEG_LINK2", "RLEG_LINK2"]
        # self.knee_indices = torch.zeros(
        #     len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        # self.base_index = self.gym.find_actor_rigid_body_handle(
        #     self.envs[0], self.khr_handles[0], feet_names[i])

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
                env_ptr, khr_asset, start_pose, "khr", i, 0, 0)  # self collision 1 to 0
            self.gym.set_actor_dof_properties(env_ptr, khr_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, khr_handle)

            # self.gym.end_aggregate(env_ptr)

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

        # for i in range(len(knee_names)):
        #     self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
        #         self.envs[0], self.khr_handles[0], knee_names[i])
        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.khr_handles[0], "BODY")

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

        # dof limits
        self.dof_lower_limit = lower_limits
        self.dof_upper_limit = upper_limits

    def pre_physics_step(self, actions):
        # actions size : (num_envs, num_actions)
        self.actions = actions.clone().to(self.device)
        # clipped by -1.0 to 1.0
        target_pos = torch.clamp(self.actions, min=-1.0, max=1.0)
        target_pos = self.action_scale * target_pos + \
            self.prev_target_pos    # action
        target_pos = dof_joint_limit_clip(
            target_pos, self.dof_lower_limit, self.dof_upper_limit)   # clipped by joiint limit
        force_tensor = set_pd_force_tensor_limit(
            self.Kp,
            self.Kd,
            target_pos,
            self.dof_pos,
            self.target_vel,
            self.dof_vel,
            self.torque_limit)

        self.force_tensor = force_tensor

        # test for zero action
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torch.zeros_like(force_tensor, dtype=torch.float, device=self.device, requires_grad=False)))

        # apply force tensor clipped by torque limit
        # self.gym.set_dof_actuation_force_tensor(
        #     self.sim, gymtorch.unwrap_tensor(force_tensor))  # apply actuator force
        # self.prev_target_pos = target_pos  # set previous target position

        if self.progress_buf[0] > self.init_dropping_step:
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(force_tensor))  # apply actuator force
            # set previous target position
            self.prev_target_pos[:] = target_pos
        else:
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(
                torch.zeros_like(force_tensor, dtype=torch.float, device=self.device, requires_grad=False)))
            self.prev_target_pos[:] = self.dof_pos[:]

        # normal PD control
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # save prev actions
        self.prev_actions[:] = self.actions
        self.prev_dof_vel[:] = self.dof_vel

        # debug
        if self.viewer and self.debug_viz:
            self.axis_visualiztion(self.feet_indices)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_khr_locomotion_reward(
            # tensors
            self.obs_buf,
            self.root_states,
            self.rb_states,
            self.dof_pos,
            self.dof_vel,
            self.dof_acc,
            self.gravity_vec,
            self.target_z_height,
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
            self.init_dropping_step,
            self.max_episode_length,
            # self.base_target_state,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # print('debug')
        # print(self.obs_buf[0][0:32])
        # print(self.obs_buf[0][32:64])
        # print(self.obs_buf[0][64:96])

        # compute acceleration
        self.dof_acc = compute_dof_acceleration(
            self.dof_vel, self.prev_dof_vel, self.dt)

        self.obs_buf[:] = compute_khr_locomotion_observations(  # tensors
            self.obs_buf[:, 2 * self.num_actions: 2 * self.num_actions + self.prev_state_buffer_step * \
                         self.num_actions * 2],
            scale_joint_pos(self.dof_pos, self.dof_lower_limit,
                            self.dof_upper_limit),
            scale_joint_pos(self.prev_target_pos,
                            self.dof_lower_limit, self.dof_upper_limit),
            self.root_states,
            self.ang_vel_scale
        )

    def reset_idx(self, env_ids):

        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        initial_roll, initial_pitch, initial_yaw = get_euler_xyz(
            self.initial_root_states[:, 3:7])
        # randomize initial quaternion states
        if self.random_initialize:
            # initial_root_euler_delta = torch_rand_float(
            #     -torch.pi / 2., torch.pi / 2., (self.num_envs, 2), device=self.device)
            # initial_roll_delta, initial_pitch_delta = initial_root_euler_delta[
            #     :, 0], initial_root_euler_delta[:, 1]

            # initial_roll_rand, initial_pitch_rand = initial_roll + \
            #     initial_roll_delta, initial_pitch + initial_pitch_delta
            # initial_root_quat_rand = quat_from_euler_xyz(
            #     initial_roll_rand, initial_pitch_rand, initial_yaw)
            # initial_root_states = self.initial_root_states.detach()
            # initial_root_states[:, 3:7] = initial_root_quat_rand
            # initial_root_states = self.initial_root_states +
            pass
        else:
            initial_root_states = self.initial_root_states
            initial_roll_rand = initial_roll
            initial_pitch_rand = initial_pitch

        positions_offset = torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        # velocities = torch_rand_float(-0.01, 0.01,
        #                               (len(env_ids), self.num_dof), device=self.device)
        velocities = 0.

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

        # reset prev target pos
        self.prev_target_pos[env_ids] = self.dof_pos[env_ids]

        # reset obs buffer for dof pos and prev target dof pos
        for i in range(self.prev_state_buffer_step + 1):
            self.obs_buf[env_ids, (2 * i) * self.num_actions: (2 + 2 * i) * self.num_actions] = \
                torch.cat((scale_joint_pos(self.dof_pos[env_ids], self.dof_lower_limit, self.dof_upper_limit),
                          scale_joint_pos(self.prev_target_pos[env_ids], self.dof_lower_limit, self.dof_upper_limit)), dim=1)

        # reset obs buffer for roll, pitch
        self.obs_buf[env_ids, self.num_actions * 2 * (self.prev_state_buffer_step + 1): self.num_actions * 2 * (
            self.prev_state_buffer_step + 1) + 2] = torch.hstack([initial_roll_rand[env_ids].unsqueeze(1), initial_pitch_rand[env_ids].unsqueeze(1)])

        # reset obs buffer for base_ang_vel
        self.obs_buf[env_ids, self.num_actions * 2 * (self.prev_state_buffer_step + 1) + 2: self.num_actions * 2 * (
            self.prev_state_buffer_step + 1) + 5] = initial_root_states[env_ids, 10:13] * self.ang_vel_scale

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
def dof_joint_limit_clip(dof_state, dof_lower_limit, dof_upper_limit):
    clipped_dof_state = torch.max(
        torch.min(dof_state, dof_upper_limit), dof_lower_limit)
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
    force_tensor = Kp * (target_pos - current_pos) + \
        Kd * (target_vel - current_vel)
    force_tensor = torch.clamp(
        force_tensor, min=-torque_limit, max=torque_limit)
    return force_tensor


@torch.jit.script
def compute_khr_locomotion_reward(
        # tensor
        obs_buf,
        root_states,
        rb_states,
        dof_pos,
        dof_vel,
        dof_acc,
        gravity_vec,
        target_z_height,
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
        init_dropping_step,
        max_episode_length,
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, int, int, int) -> Tuple[Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)

    base_pos = root_states[:, :3]
    base_quat = root_states[:, 3:7]
    # world frame, if quat_rotate_inverse(base_quat, base_lin_vel), local base
    base_lin_vel = root_states[:, 7:10]
    base_ang_vel = root_states[:, 10:13]  # world frame

    # upright reward
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
    # gravity_vec_error = torch.sum(torch.square(
    #     gravity_vec - projected_gravity), dim=1)
    # rew_gravity_vector_error = torch.exp(-gravity_vec_error) * \
    #     rew_scales["gravityVector"]
    gravity_vec_error = gravity_vec - projected_gravity
    gravity_vec_error_norm = torch.norm(gravity_vec_error, dim=1)
    rew_gravity_vector_error = torch.exp(- 20. * gravity_vec_error_norm) * \
        rew_scales["gravityVector"]

    # height reward
    base_z_height = base_pos[:, 2]
    # rew_z_height = torch.zeros_like(base_z_height)
    # rew_z_height = torch.where(base_z_height > target_z_height, rew_z_height + 1., rew_z_height) * rew_scales["zHeight"]
    rew_z_height_error = torch.square(base_z_height - target_z_height)
    rew_z_height = torch.exp(- 20. * rew_z_height_error) * \
        rew_scales["zHeight"]
    # rew_z_height = base_z_height * rew_scales["zHeight"]

    # return pos error reward (pos difference between target pos which is standing up configuration)
    dof_pos_error = torch.sum(torch.square(dof_pos - target_dof_pos), dim=1)
    rew_dof_pos_error = torch.exp(- 2. * dof_pos_error) * \
        rew_scales["posError"]

    # target pos difference reward
    # current_target_pos = obs_buf[:, num_actions: 2 * num_actions]
    # ahead_target_pos = obs_buf[:, 2 * num_actions +
    #                            num_actions: 2 * num_actions + 2 * num_actions]
    # target_pos_diff = torch.sum(torch.square(
    #     current_target_pos - ahead_target_pos), dim=1)
    # rew_target_pos_diff = torch.exp(- 2. * target_pos_diff) * \
    #     rew_scales["targetPosDiff"]
    current_target_pos = obs_buf[:, num_actions * 2 * prev_state_buffer_step +
                                 num_actions: num_actions * 2 * prev_state_buffer_step + num_actions * 2]
    ahead_target_pos = obs_buf[:, num_actions * 2 * (
        prev_state_buffer_step - 1) + num_actions: num_actions * 2 * prev_state_buffer_step]
    # target_pos_diff = torch.sum(torch.square(
    #     current_target_pos - ahead_target_pos), dim=1)
    target_pos_diff = torch.norm(current_target_pos - ahead_target_pos, dim=1)
    rew_target_pos_diff = target_pos_diff * rew_scales["targetPosDiff"]

    # foot contact
    # reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    # feet_contact = torch.norm(contact_forces[:, feet_indices, :], dim=2) > 0.5
    # only Z-direction force
    feet_contact = torch.abs(contact_forces[:, feet_indices, 2]) > 0.5
    rew_feet_contact = torch.all(
        feet_contact, dim=1) * rew_scales["feetContact"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    # power penalty
    rew_power = torch.sum(torch.square(dof_vel * torques),
                          dim=1) * rew_scales["power"]

    # action difference penalty
    actions_diff = torch.sum(torch.square(
        actions - prev_actions), dim=1)
    rew_actions_diff = actions_diff * rew_scales["actionsDiff"]

    # joint velocity penalty
    rew_dof_vel = torch.sum(torch.square(
        dof_vel), dim=1) * rew_scales["dofVelocity"]

    # joint acceleration penalty
    rew_dof_acc = torch.sum(torch.square(
        dof_acc), dim=1) * rew_scales["dofAcceleration"]

    # foot upright reward when touched ground
    feet_quats = rb_states[:, feet_indices, 3:7]
    projected_gravity_wrt_feet = quat_rotate_inverse(
        feet_quats.view(-1, 4), gravity_vec.repeat(len(feet_indices), 1)).view(-1, 3)  # project gravity vector to feet
    gravity_vec_error_wrt_feet = torch.sum(torch.square(
        gravity_vec.repeat(len(feet_indices), 1) - projected_gravity_wrt_feet), dim=1)  # gravity error wrt feet
    gravity_vec_error_wrt_feet_masked = torch.exp(- 10. * gravity_vec_error_wrt_feet).view(-1, len(
        feet_indices)) * feet_contact  # masked using feet_contact
    rew_gravity_vector_error_wrt_feet = torch.sum(
        gravity_vec_error_wrt_feet_masked, dim=1) * rew_scales["feetUpright"]

    # base contact
    base_contact = torch.norm(contact_forces[:, base_index, :], dim=1) > 0.3
    rew_base_contact = base_contact * rew_scales["bodyContact"]

    # body impulse (contact force) penalty
    contact_forces_norm_total = torch.sum(
        torch.norm(contact_forces, dim=2), dim=1)
    rew_contact_forces = contact_forces_norm_total * \
        rew_scales["contactForces"]

    # base angular velocity penalty
    # rew_base_lin_vel = torch.sum(torch.square(
    #     base_lin_vel), dim=1) * rew_scales["baseLinearVelocity"]
    rew_base_lin_vel = torch.norm(
        base_lin_vel, dim=1) * rew_scales["baseLinearVelocity"]

    # base angular velocity penalty
    # rew_base_ang_vel = torch.sum(torch.square(
    #     base_ang_vel), dim=1) * rew_scales["baseAngularVelocity"]
    rew_base_ang_vel = torch.norm(
        base_ang_vel, dim=1) * rew_scales["baseAngularVelocity"]

    # calculate total reward
    total_reward = rew_gravity_vector_error + \
        rew_gravity_vector_error_wrt_feet + \
        rew_z_height + \
        rew_dof_pos_error + \
        rew_target_pos_diff + \
        rew_feet_contact + \
        rew_base_contact + \
        rew_torque + \
        rew_power + \
        rew_dof_vel + \
        rew_dof_acc + \
        rew_base_lin_vel + \
        rew_base_ang_vel + \
        rew_actions_diff + \
        rew_contact_forces
    total_reward = torch.clip(total_reward, 0., None)

    # print("debug")
    # print("rew_gravity_vector_error", rew_gravity_vector_error[0])
    # print("rew_gravity_vector_error_wrt_feet",
    #       rew_gravity_vector_error_wrt_feet[0])
    # print("rew_z_height", rew_z_height[0])
    # print("rew_dof_pos_error", rew_dof_pos_error[0])
    # print("rew_target_pos_diff", rew_target_pos_diff[0])
    # print("rew_feet_contact", rew_feet_contact[0])
    # print("rew_base_contact", rew_base_contact[0])
    # print("rew_torque", rew_torque[0])
    # print("rew_power", rew_power[0])
    # print("rew_dof_vel", rew_dof_vel[0])
    # print("rew_dof_acc", rew_dof_acc[0])
    # print("rew_base_line_vel", rew_base_lin_vel[0])
    # print("rew_base_ang_vel", rew_base_ang_vel[0])
    # print("rew_actions_diff", rew_actions_diff[0])
    # print("rew_contact_forces", rew_contact_forces[0])
    # print('feet contact')
    # print(feet_contact[0])
    # print('base_contact')
    # print(base_contact[0])
    # print('debug')
    # print(contact_forces.shape)

    # no reward when initializing
    total_reward = total_reward * (episode_lengths > init_dropping_step)

    # reset when time out
    time_out = episode_lengths >= max_episode_length - \
        1

    reset = time_out

    # reset when base contacts after 250 steps
    reset = reset | (base_contact * (episode_lengths - 250 > 0))

    # reset when feet not contact after 300 steps
    # feet_not_contact = ~torch.any(feet_contact, dim=1)
    # reset = reset | (feet_not_contact * (episode_lengths - 300 > 0))

    # reset when other body contact after 300 steps
    # other_body_contact
    other_body_contact = torch.any(torch.norm(
        contact_forces[:, other_indices, :], dim=2) > 0.2, dim=1)
    reset = reset | (other_body_contact * (episode_lengths - 300 > 0))

    # reset when base_z_height is lower than target_z_height - 1.0 after 300 steps
    reset = reset | (base_z_height < target_z_height - 1.0) * \
        (episode_lengths - 300 > 0)

    # reset when feet is too high after 250 steps
    reset = reset | (torch.any(
        rb_states[:, feet_indices, 2] > 0.1, dim=-1) * (episode_lengths - 250 > 0))

    # reset when base_lin_vel is too high after initilizing
    # reset = reset | (torch.norm(base_lin_vel, dim=-1) > 3.0) * \
    #     (episode_lengths - init_dropping_step > 0)

    # reset when base_ang_vel is too high after initilizing
    # reset = reset | (torch.norm(base_ang_vel, dim=-1) > 15.0) * \
    #     (episode_lengths - init_dropping_step > 0)

    # reset when contact forces too high (impulse)
    # reset = reset | (torch.sum(torch.norm(contact_forces, dim=2),
    #                  dim=1) * (episode_lengths - init_dropping_step > 0))[0]

    return total_reward.detach(), reset


@ torch.jit.script
def compute_khr_locomotion_observations(prev_joint_obs_buf,
                                        dof_pos_scaled,
                                        prev_target_pos_scaled,
                                        root_states,
                                        ang_vel_scale
                                        ):

    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    base_quat = root_states[:, 3:7]
    roll, pitch, yaw = get_euler_xyz(base_quat)
    # base_ang_vel = quat_rotate_inverse(
    #     base_quat, root_states[:, 10:13]) * ang_vel_scale
    base_ang_vel = root_states[:, 10:13] * ang_vel_scale

    obs = torch.cat((
        prev_joint_obs_buf,
        dof_pos_scaled,
        prev_target_pos_scaled,
        roll.unsqueeze(1),
        pitch.unsqueeze(1),
        base_ang_vel
    ), dim=-1)
    return obs


@ torch.jit.script
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
