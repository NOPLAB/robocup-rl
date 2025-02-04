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


class RoboCupEnv(VecEnv):
    def __init__(self, num_envs, env_cfg, episode_length_s, show_viewer=False):
        super().__init__(
            num_envs=num_envs,
            observation_space=spaces.Box(
                high=1.0, low=-1.0, shape=(env_cfg["num_obs"],)
            ),
            action_space=spaces.Box(
                high=1.0, low=-1.0, shape=(env_cfg["num_actions"],)
            ),
        )

        self.num_actions = env_cfg["num_actions"]
        self.num_obs = env_cfg["num_obs"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(episode_length_s / self.dt)

        self.env_cfg = env_cfg
        self.dones = np.zeros((self.num_envs,), dtype=bool)

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
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = np.array(env_cfg["base_init_pos"], dtype=np.float32)
        self.base_init_quat = np.array(env_cfg["base_init_quat"], dtype=np.float32)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.Cylinder(
                height=env_cfg["robot_height"],
                radius=env_cfg["robot_radius"],
                pos=self.base_init_pos,
                quat=self.base_init_quat,
            )
        )

        # build
        self.scene.build(n_envs=self.num_envs)

        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

        # initialize buffers
        self.obs_buf = np.zeros((self.num_envs, self.num_obs), dtype=np.float32)
        self.actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self.episode_length_buf = np.zeros((self.num_envs,), dtype=np.int32)
        self.base_lin_vel = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.base_ang_vel = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.last_actions = np.zeros_like(self.actions)
        self.base_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.base_quat = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.base_euler = np.zeros((self.num_envs, 3), dtype=np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        clipped_actions = np.clip(
            self.actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        exec_actions = (
            self.last_actions if self.simulate_action_latency else clipped_actions
        )
        scaled_actions = exec_actions * self.env_cfg["action_scale"]

        force = [[[a[0], a[1], 0.0]] for a in scaled_actions]
        self.rigid_solver.apply_links_external_force(
            force=force,
            links_idx=[self.robot.idx],
            envs_idx=np.arange(self.num_envs),
        )

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos().cpu().numpy()
        self.base_quat[:] = self.robot.get_quat().cpu().numpy()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                np.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            )
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(
            self.robot.get_vel().cpu().numpy(), inv_base_quat
        )
        self.base_ang_vel[:] = transform_by_quat(
            self.robot.get_ang().cpu().numpy(), inv_base_quat
        )

        # update observation buffer
        self.obs_buf[:, 0] = self.base_lin_vel[:, 0]
        self.obs_buf[:, 1] = self.base_lin_vel[:, 1]

        # update last actions
        self.last_actions = self.actions

        # rewards
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        rewards += self._reward_lin_vel_x()
        rewards += self._reward_lin_vel_y()

        # dones
        dones = np.zeros((self.num_envs,), dtype=bool)
        dones |= self.episode_length_buf >= self.max_episode_length

        infos = []
        infos = [{"terminal_observation": obs} for obs in self.obs_buf]

        return (self.obs_buf, rewards, dones, infos)

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0

    def reset(self) -> VecEnvObs:
        self.reset_idx(np.arange(self.num_envs))

        return self.obs_buf

    def close(self):
        pass

    def _reward_lin_vel_x(self):
        return np.abs(self.base_lin_vel[:, 0])

    def _reward_lin_vel_y(self):
        return np.abs(self.base_lin_vel[:, 1]) * -1.0

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
