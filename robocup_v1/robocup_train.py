import genesis as gs

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from robocup_env import RoboCupEnv, get_env_cfg


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self._log_freq = 10

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            rewards = self.locals["env"].episode_reward_sums

            rewards_mean = sum(rewards) / len(rewards)

            self.logger.record("episode rewards_mean", rewards_mean)

        return True


def main():
    gs.init()

    env_cfg = get_env_cfg()

    env = RoboCupEnv(
        num_envs=4096,
        env_cfg=env_cfg,
        episode_length_s=5,
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
    ).learn(total_timesteps=20000000, callback=TensorboardCallback(), progress_bar=True)

    # Save the model
    model.save("checkpoints/ppo_robocup")


if __name__ == "__main__":
    main()
