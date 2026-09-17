from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("ResetButton-v1", max_episode_steps=100)
class ResetButtonEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    button_xy = (0.0, 0.0)
    pedestal_half = (0.06, 0.06, 0.035)
    cap_radius = 0.05
    cap_half_length = 0.015
    cap_rest_gap = 0.02
    travel = 0.02
    trigger_frac = 0.6
    spring_stiffness = 1000.0
    spring_damping = 20.0
    approach_weight = 5.0
    press_weight = 2.0
    success_bonus = 20.0
    action_penalty = 5e-4

    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        kwargs.setdefault("control_mode", "pd_ee_delta_pos")
        kwargs.setdefault("reward_mode", "dense")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.7, 0.7, 0.6], target=[0.0, 0.0, 0.15])
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.button = self._build_button()

    def _build_button(self):
        builder = self.scene.create_articulation_builder()
        builder.set_initial_pose(
            sapien.Pose(p=[self.button_xy[0], self.button_xy[1], 0.0])
        )
        base = builder.create_link_builder()
        base.set_name("base")
        base.add_box_visual(
            pose=sapien.Pose(p=[0, 0, self.pedestal_half[2]]),
            half_size=list(self.pedestal_half),
            material=sapien.render.RenderMaterial(base_color=[0.15, 0.15, 0.17, 1]),
        )
        base.add_box_collision(
            pose=sapien.Pose(p=[0, 0, self.pedestal_half[2]]),
            half_size=list(self.pedestal_half),
        )

        cap = builder.create_link_builder(base)
        cap.set_name("cap")
        cap.set_joint_name("button_joint")
        q = [0.7071068, 0.0, 0.7071068, 0.0]
        anchor_z = 2 * self.pedestal_half[2] + self.cap_rest_gap
        cap.set_joint_properties(
            type="prismatic",
            limits=[[0.0, self.travel]],
            pose_in_parent=sapien.Pose(p=[0, 0, anchor_z], q=q),
            pose_in_child=sapien.Pose(p=[0, 0, -self.cap_half_length], q=q),
            damping=self.spring_damping,
        )
        cap.add_cylinder_visual(
            pose=sapien.Pose(p=[0, 0, 0]),
            radius=self.cap_radius,
            half_length=self.cap_half_length,
            material=sapien.render.RenderMaterial(base_color=[0.85, 0.1, 0.1, 1]),
        )
        cap.add_cylinder_visual(
            pose=sapien.Pose(p=[0, 0, -self.cap_half_length - 0.01]),
            radius=0.02,
            half_length=0.02,
            material=sapien.render.RenderMaterial(base_color=[0.3, 0.3, 0.32, 1]),
        )
        cap.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0]),
            half_size=[self.cap_radius, self.cap_radius, self.cap_half_length],
        )

        button = builder.build(name="button", fix_root_link=True)
        for joint in button.get_joints():
            if "prismatic" in joint.type:
                joint.set_drive_properties(self.spring_stiffness, self.spring_damping)
                joint.set_drive_target(0.0)
                joint.set_drive_velocity_target(0.0)
        return button

    def _depression(self):
        qpos = self.button.get_qpos().reshape(self.num_envs, -1)
        return torch.clamp(qpos[:, 0], 0.0, self.travel)

    def _button_top(self):
        root_p = self.button.pose.p
        top_lift = (
            2 * self.pedestal_half[2]
            + self.cap_rest_gap
            + 2 * self.cap_half_length
            - self._depression()
        )
        return torch.stack(
            [root_p[:, 0], root_p[:, 1], root_p[:, 2] + top_lift], dim=1
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self.button.set_qpos(torch.zeros((len(env_idx), 1), device=self.device))
        dist = torch.linalg.norm(self._button_top() - self.agent.tcp_pose.p, axis=1)
        depression = self._depression()
        if not hasattr(self, "prev_dist"):
            self.prev_dist = dist.clone()
            self.prev_depression = depression.clone()
        else:
            self.prev_dist[env_idx] = dist[env_idx].clone()
            self.prev_depression[env_idx] = depression[env_idx].clone()

    def _get_obs_extra(self, info: dict):
        return dict(
            tcp_to_button=self._button_top() - self.agent.tcp_pose.p,
            button_depression=(self._depression() / self.travel).unsqueeze(-1),
        )

    def evaluate(self):
        depression = self._depression()
        return {
            "success": depression >= self.trigger_frac * self.travel,
            "button_depressed": depression > 1e-4,
            "depression": depression,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        dist = torch.linalg.norm(self._button_top() - self.agent.tcp_pose.p, axis=1)
        depression = self._depression()
        reward = self.approach_weight * (self.prev_dist - dist)
        reward += self.press_weight * (depression - self.prev_depression)
        reward += self.success_bonus * info["success"].float()
        if action is not None:
            reward -= self.action_penalty * (action**2).sum(dim=-1)
        self.prev_dist = dist.clone()
        self.prev_depression = depression.clone()
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / self.success_bonus


@register_env("ResetButton-v2", max_episode_steps=100)
class ResetButtonRandomEnv(ResetButtonEnv):
    button_xy_low = (-0.02, -0.1)
    button_xy_high = (0.14, 0.1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            low = torch.tensor(self.button_xy_low)
            high = torch.tensor(self.button_xy_high)
            xy = low + torch.rand((b, 2)) * (high - low)
            p = torch.zeros((b, 3))
            p[:, :2] = xy
            q = torch.zeros((b, 4))
            q[:, 0] = 1.0
            self.button.set_pose(Pose.create_from_pq(p, q))
        super()._initialize_episode(env_idx, options)


@register_env("ResetButton-v3", max_episode_steps=100)
class ResetButtonFingertipEnv(ResetButtonRandomEnv):
    centered_radius = 0.04

    def _centered(self):
        tcp = self.agent.tcp_pose.p
        root = self.button.pose.p
        return torch.linalg.norm(tcp[:, :2] - root[:, :2], dim=1) <= self.centered_radius

    def evaluate(self):
        depression = self._depression()
        centered = self._centered()
        return {
            "success": (depression >= self.trigger_frac * self.travel) & centered,
            "button_depressed": depression > 1e-4,
            "centered": centered,
            "depression": depression,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        dist = torch.linalg.norm(self._button_top() - self.agent.tcp_pose.p, axis=1)
        depression = self._depression()
        centered = info["centered"].float()
        reward = self.approach_weight * (self.prev_dist - dist)
        reward += self.press_weight * (depression - self.prev_depression) * centered
        reward += self.success_bonus * info["success"].float()
        if action is not None:
            reward -= self.action_penalty * (action**2).sum(dim=-1)
        self.prev_dist = dist.clone()
        self.prev_depression = depression.clone()
        return reward


@register_env("ResetButton-v4", max_episode_steps=100)
class ResetButtonPenaltyEnv(ResetButtonFingertipEnv):
    centered_radius = 0.03
    bad_press_weight = 6.0

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        dist = torch.linalg.norm(self._button_top() - self.agent.tcp_pose.p, axis=1)
        depression = self._depression()
        press_delta = depression - self.prev_depression
        centered = info["centered"].float()
        reward = self.approach_weight * (self.prev_dist - dist)
        reward += self.press_weight * press_delta * centered
        reward -= self.bad_press_weight * press_delta * (1.0 - centered)
        reward += self.success_bonus * info["success"].float()
        if action is not None:
            reward -= self.action_penalty * (action**2).sum(dim=-1)
        self.prev_dist = dist.clone()
        self.prev_depression = depression.clone()
        return reward


@register_env("ResetButton-v5", max_episode_steps=100)
class ResetButtonJerkEnv(ResetButtonFingertipEnv):
    jerk_weight = 0.02

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if hasattr(self, "prev_action"):
            self.prev_action[env_idx] = 0.0

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        if not hasattr(self, "prev_action"):
            self.prev_action = torch.zeros_like(action)
        dist = torch.linalg.norm(self._button_top() - self.agent.tcp_pose.p, axis=1)
        depression = self._depression()
        centered = info["centered"].float()
        jerk = ((action - self.prev_action) ** 2).sum(dim=-1)
        reward = self.approach_weight * (self.prev_dist - dist)
        reward += self.press_weight * (depression - self.prev_depression) * centered
        reward += self.success_bonus * info["success"].float()
        reward -= self.jerk_weight * jerk
        reward -= self.action_penalty * (action**2).sum(dim=-1)
        self.prev_dist = dist.clone()
        self.prev_depression = depression.clone()
        self.prev_action = action.detach().clone()
        return reward



@register_env("ResetButton-v6", max_episode_steps=200)
class ResetButtonOrderEnv(ResetButtonEnv):
    button_xy_low = (-0.02, -0.11)
    button_xy_high = (0.14, 0.11)
    button_min_sep = 0.15
    blue_bonus = 20.0
    red_bonus = 5.0
    red_approach_weight = 2.0
    red_approach_weight_after_blue = 5.0

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.button = self._build_button_at(
            "button_blue", [0.1, 0.35, 0.9, 1], [0.0, -0.11]
        )
        self.button_red = self._build_button_at(
            "button_red", [0.9, 0.15, 0.15, 1], [0.12, 0.11]
        )

    def _build_button_at(self, name, color, xy):
        builder = self.scene.create_articulation_builder()
        builder.set_initial_pose(sapien.Pose(p=[xy[0], xy[1], 0.0]))
        base = builder.create_link_builder()
        base.set_name("base")
        base.add_box_visual(
            pose=sapien.Pose(p=[0, 0, self.pedestal_half[2]]),
            half_size=list(self.pedestal_half),
            material=sapien.render.RenderMaterial(base_color=[0.15, 0.15, 0.17, 1]),
        )
        base.add_box_collision(
            pose=sapien.Pose(p=[0, 0, self.pedestal_half[2]]),
            half_size=list(self.pedestal_half),
        )
        cap = builder.create_link_builder(base)
        cap.set_name("cap")
        cap.set_joint_name("button_joint")
        q = [0.7071068, 0.0, 0.7071068, 0.0]
        anchor_z = 2 * self.pedestal_half[2] + self.cap_rest_gap
        cap.set_joint_properties(
            type="prismatic",
            limits=[[0.0, self.travel]],
            pose_in_parent=sapien.Pose(p=[0, 0, anchor_z], q=q),
            pose_in_child=sapien.Pose(p=[0, 0, -self.cap_half_length], q=q),
            damping=self.spring_damping,
        )
        cap.add_cylinder_visual(
            pose=sapien.Pose(p=[0, 0, 0]),
            radius=self.cap_radius,
            half_length=self.cap_half_length,
            material=sapien.render.RenderMaterial(base_color=color),
        )
        cap.add_cylinder_visual(
            pose=sapien.Pose(p=[0, 0, -self.cap_half_length - 0.01]),
            radius=0.02,
            half_length=0.02,
            material=sapien.render.RenderMaterial(base_color=[0.3, 0.3, 0.32, 1]),
        )
        cap.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0]),
            half_size=[self.cap_radius, self.cap_radius, self.cap_half_length],
        )
        button = builder.build(name=name, fix_root_link=True)
        for joint in button.get_joints():
            if "prismatic" in joint.type:
                joint.set_drive_properties(self.spring_stiffness, self.spring_damping)
                joint.set_drive_target(0.0)
                joint.set_drive_velocity_target(0.0)
        return button

    def _depression_red(self):
        qpos = self.button_red.get_qpos().reshape(self.num_envs, -1)
        return torch.clamp(qpos[:, 0], 0.0, self.travel)

    def _button_top_red(self):
        root_p = self.button_red.pose.p
        top_lift = (
            2 * self.pedestal_half[2]
            + self.cap_rest_gap
            + 2 * self.cap_half_length
            - self._depression_red()
        )
        return torch.stack(
            [root_p[:, 0], root_p[:, 1], root_p[:, 2] + top_lift], dim=1
        )

    def _sample_layout(self, b):
        low = torch.tensor(self.button_xy_low)
        high = torch.tensor(self.button_xy_high)
        corners = torch.tensor(
            [
                [low[0], low[1]],
                [low[0], high[1]],
                [high[0], low[1]],
                [high[0], high[1]],
            ]
        )
        blue = low + torch.rand((b, 2)) * (high - low)
        red = low + torch.rand((b, 2)) * (high - low)
        for i in range(b):
            for _ in range(500):
                if torch.linalg.norm(blue[i] - red[i]) >= self.button_min_sep:
                    break
                red[i] = low + torch.rand(2) * (high - low)
            if torch.linalg.norm(blue[i] - red[i]) < self.button_min_sep:
                red[i] = corners[torch.argmax(torch.linalg.norm(corners - blue[i], dim=1))]
        return blue, red

    def _set_pose_xy(self, articulation, env_idx, xy):
        raw = articulation.pose.raw_pose.clone()
        raw[env_idx, 0] = xy[:, 0].to(raw.dtype)
        raw[env_idx, 1] = xy[:, 1].to(raw.dtype)
        raw[env_idx, 2] = 0.0
        raw[env_idx, 3] = 1.0
        raw[env_idx, 4] = 0.0
        raw[env_idx, 5] = 0.0
        raw[env_idx, 6] = 0.0
        articulation.set_pose(Pose.create_from_pq(raw[:, :3], raw[:, 3:]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            blue, red = self._sample_layout(b)
            self._set_pose_xy(self.button, env_idx, blue)
            self._set_pose_xy(self.button_red, env_idx, red)
            self.button.set_qpos(torch.zeros((b, 1), device=self.device))
            self.button_red.set_qpos(torch.zeros((b, 1), device=self.device))
        dist_blue = torch.linalg.norm(
            self._button_top() - self.agent.tcp_pose.p, axis=1
        )
        dist_red = torch.linalg.norm(
            self._button_top_red() - self.agent.tcp_pose.p, axis=1
        )
        if not hasattr(self, "prev_dist"):
            self.prev_dist = dist_blue.clone()
            self.prev_dist_red = dist_red.clone()
            self._blue_pressed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._red_pressed = self._blue_pressed.clone()
        self.prev_dist[env_idx] = dist_blue[env_idx].clone()
        self.prev_dist_red[env_idx] = dist_red[env_idx].clone()
        self._blue_pressed[env_idx] = False
        self._red_pressed[env_idx] = False

    def evaluate(self):
        dep_blue = self._depression()
        dep_red = self._depression_red()
        blue_now = dep_blue >= self.trigger_frac * self.travel
        red_now = dep_red >= self.trigger_frac * self.travel
        blue_done = self._blue_pressed
        newly_blue = blue_now & ~blue_done
        newly_red = red_now & ~self._red_pressed
        order_success = red_now & blue_done
        red_first = red_now & ~blue_done
        self._blue_pressed = blue_done | blue_now
        self._red_pressed = self._red_pressed | red_now
        return {
            "success": red_now,
            "order_success": order_success,
            "red_first": red_first,
            "newly_blue": newly_blue,
            "newly_red": newly_red,
            "blue_now": blue_now,
            "red_now": red_now,
            "blue_done": blue_done,
        }

    def _get_obs_extra(self, info: dict):
        tcp = self.agent.tcp_pose.p
        return dict(
            tcp_to_blue=self._button_top() - tcp,
            tcp_to_red=self._button_top_red() - tcp,
            dep_blue=(self._depression() / self.travel).unsqueeze(-1),
            dep_red=(self._depression_red() / self.travel).unsqueeze(-1),
            blue_done=self._blue_pressed.float().unsqueeze(-1),
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp = self.agent.tcp_pose.p
        dist_blue = torch.linalg.norm(self._button_top() - tcp, axis=1)
        dist_red = torch.linalg.norm(self._button_top_red() - tcp, axis=1)
        blue_done = info["blue_done"]
        reward = self.approach_weight * (self.prev_dist - dist_blue)
        red_weight = torch.where(
            blue_done,
            torch.full_like(dist_red, self.red_approach_weight_after_blue),
            torch.full_like(dist_red, self.red_approach_weight),
        )
        reward += red_weight * (self.prev_dist_red - dist_red)
        reward += self.blue_bonus * info["newly_blue"].float()
        reward += self.red_bonus * info["newly_red"].float()
        if action is not None:
            reward -= self.action_penalty * (action**2).sum(dim=-1)
        self.prev_dist = dist_blue.clone()
        self.prev_dist_red = dist_red.clone()
        return reward


@register_env("ResetButton-v7", max_episode_steps=200)
class ResetButtonStrictOrderEnv(ResetButtonOrderEnv):
    button_min_sep = 0.18

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp = self.agent.tcp_pose.p
        dist_blue = torch.linalg.norm(self._button_top() - tcp, axis=1)
        dist_red = torch.linalg.norm(self._button_top_red() - tcp, axis=1)
        blue_done = info["blue_done"]
        reward = self.approach_weight * (self.prev_dist - dist_blue)
        red_weight = torch.where(
            blue_done,
            torch.full_like(dist_red, self.red_approach_weight_after_blue),
            torch.full_like(dist_red, self.red_approach_weight),
        )
        reward += red_weight * (self.prev_dist_red - dist_red)
        reward += self.blue_bonus * info["newly_blue"].float()
        reward += (
            self.red_bonus
            * info["newly_red"].float()
            * info["blue_done"].float()
        )
        if action is not None:
            reward -= self.action_penalty * (action**2).sum(dim=-1)
        self.prev_dist = dist_blue.clone()
        self.prev_dist_red = dist_red.clone()
        return reward


@register_env("ResetButton-v8", max_episode_steps=200)
class ResetButtonFingertipOrderEnv(ResetButtonStrictOrderEnv):
    centered_radius = 0.04

    def _centered_blue(self):
        tcp = self.agent.tcp_pose.p
        root = self.button.pose.p
        return torch.linalg.norm(tcp[:, :2] - root[:, :2], dim=1) <= self.centered_radius

    def _centered_red(self):
        tcp = self.agent.tcp_pose.p
        root = self.button_red.pose.p
        return torch.linalg.norm(tcp[:, :2] - root[:, :2], dim=1) <= self.centered_radius

    def evaluate(self):
        dep_blue = self._depression()
        dep_red = self._depression_red()
        centered_blue = self._centered_blue()
        centered_red = self._centered_red()
        blue_now = (dep_blue >= self.trigger_frac * self.travel) & centered_blue
        red_now = (dep_red >= self.trigger_frac * self.travel) & centered_red
        blue_done = self._blue_pressed
        newly_blue = blue_now & ~blue_done
        newly_red = red_now & ~self._red_pressed
        order_success = red_now & blue_done
        red_first = red_now & ~blue_done
        self._blue_pressed = blue_done | blue_now
        self._red_pressed = self._red_pressed | red_now
        return {
            "success": red_now,
            "order_success": order_success,
            "red_first": red_first,
            "newly_blue": newly_blue,
            "newly_red": newly_red,
            "blue_now": blue_now,
            "red_now": red_now,
            "blue_done": blue_done,
            "centered_blue": centered_blue,
            "centered_red": centered_red,
        }


@register_env("ResetButton-v9", max_episode_steps=250)
class ResetButtonStagedFingertipOrderEnv(ResetButtonFingertipOrderEnv):
    press_shaping_weight = 2.0

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if hasattr(self, "prev_dep_blue"):
            self.prev_dep_blue[env_idx] = self._depression()[env_idx].detach()
            self.prev_dep_red[env_idx] = self._depression_red()[env_idx].detach()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        if not hasattr(self, "prev_dep_blue"):
            self.prev_dep_blue = self._depression().detach().clone()
            self.prev_dep_red = self._depression_red().detach().clone()
        tcp = self.agent.tcp_pose.p
        dist_blue = torch.linalg.norm(self._button_top() - tcp, axis=1)
        dist_red = torch.linalg.norm(self._button_top_red() - tcp, axis=1)
        blue_done = info["blue_done"]
        dep_blue = self._depression()
        dep_red = self._depression_red()
        centered_blue = info["centered_blue"].float()
        centered_red = info["centered_red"].float()
        blue_weight = torch.where(
            blue_done, torch.zeros_like(dist_blue), torch.full_like(dist_blue, self.approach_weight)
        )
        red_weight = torch.where(
            blue_done,
            torch.full_like(dist_red, self.red_approach_weight_after_blue),
            torch.full_like(dist_red, self.red_approach_weight),
        )
        reward = blue_weight * (self.prev_dist - dist_blue)
        reward += red_weight * (self.prev_dist_red - dist_red)
        reward += (
            self.press_shaping_weight
            * (dep_blue - self.prev_dep_blue)
            * centered_blue
            * (~blue_done).float()
        )
        reward += (
            self.press_shaping_weight
            * (dep_red - self.prev_dep_red)
            * centered_red
            * blue_done.float()
        )
        reward += self.blue_bonus * info["newly_blue"].float()
        reward += (
            self.red_bonus * info["newly_red"].float() * info["blue_done"].float()
        )
        if action is not None:
            reward -= self.action_penalty * (action**2).sum(dim=-1)
        self.prev_dist = dist_blue.clone()
        self.prev_dist_red = dist_red.clone()
        self.prev_dep_blue = dep_blue.detach().clone()
        self.prev_dep_red = dep_red.detach().clone()
        return reward


@register_env("ResetButton-v10", max_episode_steps=250)
class ResetButtonCurriculumEnv(ResetButtonStagedFingertipOrderEnv):
    centered_radius = 0.05
    red_approach_weight_after_blue = 10.0
    press_shaping_weight = 4.0
    blue_start_frac = 0.3

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        start_blue = torch.rand(b) < self.blue_start_frac
        if start_blue.any():
            idx = env_idx[start_blue]
            qpos = self.button.get_qpos().clone()
            qpos[idx] = 0.015
            self.button.set_qpos(qpos)
            self._blue_pressed[idx] = True
            if not hasattr(self, "prev_dep_blue"):
                self.prev_dep_blue = self._depression().detach().clone()
                self.prev_dep_red = self._depression_red().detach().clone()
            self.prev_dep_blue[idx] = self._depression()[idx].detach()


@register_env("ResetButton-v11", max_episode_steps=250)
class ResetButtonStrongCurriculumEnv(ResetButtonCurriculumEnv):
    blue_start_frac = 0.7
    red_approach_weight_after_blue = 12.0
    press_shaping_weight = 6.0


@register_env("ResetButton-v12", max_episode_steps=250)
class ResetButtonBalancedCurriculumEnv(ResetButtonStrongCurriculumEnv):
    blue_start_frac = 0.5


@register_env("ResetButton-v13", max_episode_steps=250)
class ResetButtonStrictRewardOrderEnv(ResetButtonBalancedCurriculumEnv):
    red_bonus_after_blue = 20.0
    red_penalty_before_blue = 5.0
    red_approach_weight = 0.0

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        if not hasattr(self, "prev_dep_blue"):
            self.prev_dep_blue = self._depression().detach().clone()
            self.prev_dep_red = self._depression_red().detach().clone()
        tcp = self.agent.tcp_pose.p
        dist_blue = torch.linalg.norm(self._button_top() - tcp, axis=1)
        dist_red = torch.linalg.norm(self._button_top_red() - tcp, axis=1)
        blue_done = info["blue_done"]
        dep_blue = self._depression()
        dep_red = self._depression_red()
        centered_blue = info["centered_blue"].float()
        centered_red = info["centered_red"].float()
        blue_weight = torch.where(
            blue_done,
            torch.zeros_like(dist_blue),
            torch.full_like(dist_blue, self.approach_weight),
        )
        red_weight = torch.where(
            blue_done,
            torch.full_like(dist_red, self.red_approach_weight_after_blue),
            torch.full_like(dist_red, self.red_approach_weight),
        )
        reward = blue_weight * (self.prev_dist - dist_blue)
        reward += red_weight * (self.prev_dist_red - dist_red)
        reward += (
            self.press_shaping_weight
            * (dep_blue - self.prev_dep_blue)
            * centered_blue
            * (~blue_done).float()
        )
        reward += (
            self.press_shaping_weight
            * (dep_red - self.prev_dep_red)
            * centered_red
            * blue_done.float()
        )
        reward += self.blue_bonus * info["newly_blue"].float()
        red_bonus = torch.where(
            blue_done,
            torch.full_like(dist_red, self.red_bonus_after_blue),
            torch.full_like(dist_red, -self.red_penalty_before_blue),
        )
        reward += red_bonus * info["newly_red"].float()
        if action is not None:
            reward -= self.action_penalty * (action**2).sum(dim=-1)
        self.prev_dist = dist_blue.clone()
        self.prev_dist_red = dist_red.clone()
        self.prev_dep_blue = dep_blue.detach().clone()
        self.prev_dep_red = dep_red.detach().clone()
        return reward
