import genesis as gs

from stable_baselines3 import PPO

from robocup_env import RoboCupEnv


def get_env_cfg():
    return {
        "num_actions": 2,
        "num_obs": 2,
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [0.0, 0.0, 0.0, 1.0],
        "clip_actions": 1.0,
        "action_scale": 100.0,
        "robot_height": 0.2,
        "robot_radius": 0.2,
    }


def main():
    gs.init()

    env_cfg = get_env_cfg()

    env = RoboCupEnv(
        num_envs=1,
        env_cfg=env_cfg,
        episode_length_s=10,
        show_viewer=True,
    )

    # Load the model
    model = PPO.load(path="checkpoints/ppo_robocup")

    obs = env.reset()
    for _ in range(1000):
        actions, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, done, _ = env.step(actions=actions)

        print(f"actions: {actions}")
        print(f"obs: {obs}")
        print(f"reward: {reward}")
        print(f"done: {done}")

        if done:
            env.reset()


if __name__ == "__main__":
    main()
