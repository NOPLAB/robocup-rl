import gymnasium as gym
from gymnasium import spaces
import math
import numpy as np
import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from typing import Any
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
import torch


def get_env_cfg():
    return {
        "num_actions": 2,
        "num_obs": 2,
        "base_init_pos": [-1.0, 0.0, 0.06],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "clip_actions": 1.0,
        "action_scale": 100.0,
        "action_scale_noise_min": 0.8,
        "action_scale_noise_max": 1.2,
        "robot": {
            "height": 0.1,
            "radius": 0.2,
        },
        "ball": {
            "radius": 0.05,
            "pos": [0.0, 0.0, 0.05],
        },
        "walls": [
            # side walls
            {
                "size": [2.19, 0.01, 0.2],
                "pos": [0.0, 1.58 / 2, 0.1],
            },
            {
                "size": [2.19, 0.01, 0.2],
                "pos": [0.0, -1.58 / 2, 0.1],
            },
            # next to goal walls
            {
                "size": [0.01, (1.58 - 0.8) / 2, 0.2],
                "pos": [2.19 / 2, (1.58 - 0.8) * 0.75, 0.1],
            },
            {
                "size": [0.01, (1.58 - 0.8) / 2, 0.2],
                "pos": [-2.19 / 2, (1.58 - 0.8) * 0.75, 0.1],
            },
            # next to goal walls
            {
                "size": [0.01, (1.58 - 0.8) / 2, 0.2],
                "pos": [2.19 / 2, -(1.58 - 0.8) * 0.75, 0.1],
            },
            {
                "size": [0.01, (1.58 - 0.8) / 2, 0.2],
                "pos": [-2.19 / 2, -(1.58 - 0.8) * 0.75, 0.1],
            },
        ],
    }


class RoboCupEnv(VecEnv):
    scene: gs.Scene

    def __init__(self, num_envs, env_cfg, episode_length_s, show_viewer=False):
        super().__init__(
            num_envs=num_envs,
            observation_space=spaces.Box(
                high=1.0, low=-1.0, shape=(env_cfg["num_obs"],)
            ),
            action_space=spaces.Box(
                high=env_cfg["clip_actions"],
                low=-env_cfg["clip_actions"],
                shape=(env_cfg["num_actions"],),
            ),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = env_cfg["num_actions"]
        self.num_obs = env_cfg["num_obs"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(episode_length_s / self.dt)

        self.env_cfg = env_cfg

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        (
            self.scene.add_entity(
                morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
                material=gs.materials.Rigid(friction=1e-2),
            ),
        )

        # add robot
        self.base_init_pos = torch.tensor(
            env_cfg["base_init_pos"], dtype=torch.float32, device=self.device
        )
        self.base_init_quat = torch.tensor(
            env_cfg["base_init_quat"], dtype=torch.float32, device=self.device
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            morph=gs.morphs.Cylinder(
                height=env_cfg["robot"]["height"],
                radius=env_cfg["robot"]["radius"],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
            material=gs.materials.Rigid(rho=1400, friction=1e-2),
            surface=gs.surfaces.Plastic(),
        )

        # add ball
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=env_cfg["ball"]["radius"],
                pos=env_cfg["ball"]["pos"],
            )
        )

        # add walls
        # for wall in env_cfg["walls"]:
        #     self.scene.add_entity(
        #         gs.morphs.Box(
        #             size=wall["size"],
        #             pos=wall["pos"],
        #             fixed=True,
        #         )
        #     )

        # build
        self.scene.build(n_envs=self.num_envs)

        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

        # initialize buffers
        self.obs_buf = np.zeros((self.num_envs, self.num_obs), dtype=np.float32)
        self.actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self.last_actions = np.zeros_like(self.actions)
        self.action_scale = env_cfg["action_scale"]
        self.action_scale_noise_min = env_cfg["action_scale_noise_min"]
        self.action_scale_noise_max = env_cfg["action_scale_noise_max"]
        self.action_scale_noise = np.random.uniform(
            self.action_scale_noise_min,
            self.action_scale_noise_max,
            size=(self.num_envs, 1),
        )
        self.episode_reward_sums = np.zeros((self.num_envs,), dtype=np.float32)

        self.episode_length_buf = torch.zeros(
            (self.num_envs,), dtype=torch.int32, device=self.device
        )
        self.dones = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.rewards = torch.zeros(
            (self.num_envs,), dtype=torch.float32, device=self.device
        )

        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.base_pos = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.base_euler = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        exec_actions *= self.action_scale
        exec_actions *= self.action_scale_noise

        force = [[[a[0], a[1], 0.0]] for a in exec_actions]
        self.rigid_solver.apply_links_external_force(
            force=force,
            links_idx=[self.robot.idx],
            envs_idx=np.arange(self.num_envs),
        )

        self.lock_robot_quat(np.arange(self.num_envs))

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            )
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)

        # update observation buffer
        # vector about ball direction
        ball_vec = self.ball.get_pos() - self.base_pos
        ball_norm = ball_vec / torch.linalg.norm(ball_vec, axis=1, keepdims=True)

        self.obs_buf[:, 0] = ball_norm[:, 0].cpu().numpy()
        self.obs_buf[:, 1] = ball_norm[:, 1].cpu().numpy()

        # update last actions
        self.last_actions = self.actions

        # rewards
        self.rewards[:] = 0.0
        self.rewards += self._reward_every_step()
        self.rewards += self._reward_goal()
        # self.rewards += self._reward_touch_ball()
        self.rewards += self._reward_near_ball()
        # self.rewards += self._reward_leave_lerning_space_x()
        # self.rewards += self._reward_leave_lerning_space_y()

        rewards_cpu = self.rewards.cpu().numpy()

        self.episode_reward_sums += rewards_cpu

        # dones
        self.dones[:] = False
        self.dones |= self.episode_length_buf >= self.max_episode_length
        self.dones |= self._reward_goal() != 0.0
        # self.dones |= self._reward_touch_ball() != 0.0
        # self.dones |= self._reward_leave_lerning_space_x() != 0.0
        # self.dones |= self._reward_leave_lerning_space_y() != 0.0

        dones_cpu = self.dones.cpu().numpy()

        # reset done envs
        self.reset_idx(self.dones.nonzero(as_tuple=False).flatten().cpu().numpy())

        infos = []
        infos = [{"terminal_observation": obs} for obs in self.obs_buf]

        return (
            self.obs_buf,
            rewards_cpu,
            dones_cpu,
            infos,
        )

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = (
            torch.from_numpy(
                np.random.uniform(
                    low=[-2.19 / 2 + 0.3, -1.58 / 2 + 0.3, 0.06],
                    high=[2.19 / 2 - 0.3, 1.58 / 2 - 0.3, 0.06],
                    size=(len(envs_idx), 3),
                )
            )
            .to(torch.float32)
            .cuda()
        )
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0

        # reset ball
        ball_pos = np.random.uniform(
            low=[-2.19 / 2 + 0.3, -1.58 / 2 + 0.3, 0.1],
            high=[2.19 / 2 - 0.3, 1.58 / 2 - 0.3, 0.1],
            size=(len(envs_idx), 3),
        )
        self.ball.set_pos(ball_pos, zero_velocity=True, envs_idx=envs_idx)

        # reset random force scale
        self.action_scale_noise[envs_idx] = np.random.uniform(
            self.action_scale_noise_min,
            self.action_scale_noise_max,
            size=(len(envs_idx), 1),
        )

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.episode_reward_sums[envs_idx] = 0.0

    def reset(self) -> VecEnvObs:
        self.reset_idx(np.arange(self.num_envs))

        return self.obs_buf

    def close(self):
        pass

    def _reward_every_step(self):
        return -0.02

    def _reward_goal(self):
        ball_pos = self.ball.get_pos()
        return torch.where(ball_pos[:, 0] > 2.19 / 2, 5.0, 0.0)

    def _reward_touch_ball(self):
        ball_pos = self.ball.get_pos()
        dist = torch.linalg.norm(ball_pos - self.base_pos, axis=1)
        return torch.where(dist < 0.3, 10.0, 0.0)

    def _reward_near_ball(self):
        ball_pos = self.ball.get_pos()
        dist = torch.linalg.norm(ball_pos - self.base_pos, axis=1)
        return (1.0 / dist) / 50.0

    def _reward_leave_lerning_space_x(self):
        return torch.where(
            torch.abs(self.base_pos[:, 0]) > 2.19 / 2,
            -5.0,
            0.0,
        )

    def _reward_leave_lerning_space_y(self):
        return torch.where(
            torch.abs(self.base_pos[:, 1]) > 1.58 / 2,
            -5.0,
            0.0,
        )

    def lock_robot_quat(self, envs_idx):
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_quat(
            self.base_quat[envs_idx],
            zero_velocity=False,
            envs_idx=envs_idx,
        )

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [env_i.get_wrapper_attr(attr_name) for env_i in target_envs]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> list[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            env_i.get_wrapper_attr(method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> list[bool]:
        pass

    def _get_target_envs(self, indices: VecEnvIndices) -> list[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
