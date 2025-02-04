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
        num_envs=4096,
        env_cfg=env_cfg,
        episode_length_s=10,
        show_viewer=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=8192,
        verbose=1,
        device="cuda",
        tensorboard_log="logs/robocup",
    ).learn(total_timesteps=20000000, progress_bar=True)

    # Save the model
    model.save("checkpoints/ppo_robocup")


if __name__ == "__main__":
    main()
