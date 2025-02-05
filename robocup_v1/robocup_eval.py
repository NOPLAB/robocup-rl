import genesis as gs

import numpy as np
from stable_baselines3 import PPO

from robocup_env import RoboCupEnv, get_env_cfg


def main():
    gs.init()

    env_cfg = get_env_cfg()

    env = RoboCupEnv(
        num_envs=1,
        env_cfg=env_cfg,
        episode_length_s=5,
        show_viewer=True,
    )

    # Load the model
    model = PPO.load(path="checkpoints/ppo_robocup")

    obs = env.reset()
    for _ in range(1000):
        actions, _ = model.predict(observation=obs[0], deterministic=True)
        obs, reward, done, _ = env.step(actions=np.array([actions]))

        print(f"actions: {actions}")
        print(f"obs: {obs[0]}")
        print(f"reward: {reward[0]}")
        print(f"done: {done[0]}")

        if done:
            env.reset()


if __name__ == "__main__":
    main()
